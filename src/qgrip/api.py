"""Thin FastAPI adapter over workflow services."""

from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from qgrip.artifacts import discover_artifacts, load_calibration, subject_root
from qgrip.domain import (
    JobState,
    JobStatus,
    ModelName,
    QGripProfile,
    SGTCommand,
    SGTRequest,
    TrainingRequest,
)
from qgrip.errors import ArtifactError, QGripError
from qgrip.profiles import load_profile
from qgrip.streaming import check_streamer_device
from qgrip.workflows import WorkflowCoordinator


def notification_for(status: JobStatus) -> dict[str, object] | None:
    """Derive a discrete, ignorable toast from a terminal job transition.

    The authoritative ``status`` channel always reflects the current state;
    notifications are ephemeral courtesy events the dashboard may drop (for
    example when the tab is hidden) without losing correctness.
    """
    kind = status.kind or "job"
    if status.state == JobState.COMPLETED:
        return {"kind": kind, "level": "success", "message": f"{kind} completed"}
    if status.state == JobState.FAILED:
        return {"kind": kind, "level": "error", "message": status.message or f"{kind} failed"}
    if status.state == JobState.CANCELLED:
        return {"kind": kind, "level": "warning", "message": f"{kind} cancelled"}
    return None


class WireModel(BaseModel):
    """Base for strict HTTP request bodies that reject undeclared JSON fields."""

    model_config = ConfigDict(extra="forbid")


class SubjectWire(WireModel):
    """Request body containing a subject identifier and target-mode toggle."""

    subject: str = Field(min_length=1, max_length=128)
    discrete: bool = False


class SGTWire(SubjectWire):
    """SGT request with automatic or operator-gated presentation progression."""

    auto: bool = True


class ExportWire(WireModel):
    """Request naming the authoritative capture to project into Parquet."""

    capture: str


class TrainingWire(SubjectWire):
    """Request selecting optional datasets and a classifier architecture."""

    inputs: list[str] = []
    model: ModelName | None = None


class CalibrationWire(WireModel):
    """Request forwarded to the independent Handi calibration-jog API."""

    joint: str
    delta: float


class InferenceWire(WireModel):
    """Request selecting a checkpoint or adjacent ONNX model for live inference."""

    model: str


def create_app(
    profile: QGripProfile | str | Path,
    *,
    token: str | None = None,
    coordinator: WorkflowCoordinator | None = None,
) -> FastAPI:
    """Create the token-protected dashboard adapter over shared workflow services."""
    current = load_profile(profile) if isinstance(profile, (str, Path)) else profile
    launch_token = token or secrets.token_urlsafe(24)
    owner = coordinator or WorkflowCoordinator()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Publish application state and close the workflow owner at shutdown."""
        app.state.coordinator = owner
        app.state.profile = current
        app.state.token = launch_token
        yield
        owner.close()

    app = FastAPI(title="QGrip API", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_limits(request: Request, call_next):
        """Reject HTTP request bodies larger than the documented one-MiB limit."""
        if int(request.headers.get("content-length", "0") or 0) > 1024 * 1024:
            return JSONResponse(
                {"error": {"code": "request_too_large", "message": "request exceeds 1 MiB"}}, 413
            )
        return await call_next(request)

    @app.exception_handler(QGripError)
    async def domain_error(_request: Request, exc: QGripError):
        """Translate safe domain failures to the API's structured HTTP 409 wire form."""
        return JSONResponse({"error": {"code": type(exc).__name__, "message": str(exc)}}, 409)

    def authorize(x_qgrip_token: Annotated[str | None, Header()] = None) -> None:
        """Require the ephemeral launch token using constant-time comparison."""
        if not x_qgrip_token or not secrets.compare_digest(x_qgrip_token, launch_token):
            raise HTTPException(401, "missing or invalid QGrip launch token")

    protected = [Depends(authorize)]

    @app.get("/api/v1/bootstrap", dependencies=protected)
    def bootstrap() -> dict[str, object]:
        """Return profile-derived UI configuration values."""
        return {
            "api_version": 1,
            "profile": str(current.path),
            "device": current.device.kind,
            "gestures": current.sgt.gestures,
            "models": list(ModelName),
            "proportional": current.sgt.proportional,
            "activation_tolerance": current.sgt.activation_tolerance,
        }

    @app.get("/api/v1/doctor", dependencies=protected)
    def doctor() -> dict[str, object]:
        """Probe the configured acquisition device through its shared adapter."""
        return check_streamer_device(current.device, current.acquisition)

    @app.get("/api/v1/artifacts", dependencies=protected)
    def artifacts(subject: str | None = None) -> dict[str, object]:
        """List known artifacts globally or within one subject directory."""
        calibration_ready = False
        if subject is not None:
            calibration = subject_root(current, subject) / "calibration.json"
            if calibration.exists():
                try:
                    load_calibration(calibration, current)
                    calibration_ready = True
                except ArtifactError:
                    pass
        return {
            "artifacts": [str(path) for path in discover_artifacts(current, subject)],
            "calibration_ready": calibration_ready,
        }

    @app.post("/api/v1/sgt/start", dependencies=protected)
    def start_sgt(body: SGTWire) -> dict[str, object]:
        """Start the process's exclusive screen-guided capture workflow."""
        return asdict(
            owner.start_sgt(SGTRequest(body.subject, current, not body.discrete, body.auto))
        )

    @app.post("/api/v1/sgt/calibration/start", dependencies=protected)
    def start_calibration(body: SubjectWire) -> dict[str, object]:
        """Start the subject-specific activation calibration workflow."""
        return asdict(owner.start_calibration(body.subject, current))

    @app.post("/api/v1/sgt/command", dependencies=protected)
    def command_sgt(command: SGTCommand) -> dict[str, object]:
        """Forward one interactive command to the active SGT workflow."""
        owner.send_sgt_command(command)
        return {"accepted": True, "command": command}

    @app.get("/api/v1/sgt/status", dependencies=protected)
    @app.get("/api/v1/sgt/calibration/status", dependencies=protected)
    @app.get("/api/v1/export/status", dependencies=protected)
    @app.get("/api/v1/training/status", dependencies=protected)
    @app.get("/api/v1/inference/status", dependencies=protected)
    def status() -> dict[str, object]:
        """Return the coordinator's authoritative latest job snapshot."""
        return asdict(owner.status) if owner.status else {"state": JobState.IDLE}

    def _status_payload(status: JobStatus | None) -> str:
        """Encode a complete status snapshot for a Server-Sent Event frame."""
        snapshot = asdict(status) if status else {"state": JobState.IDLE}
        return json.dumps(snapshot, default=str)

    @app.get("/api/v1/stream", include_in_schema=False)
    async def stream(request: Request, token: str | None = None) -> StreamingResponse:
        """Open one authenticated status and notification SSE stream."""
        # EventSource cannot send custom headers, so the launch token arrives as a query.
        if not token or not secrets.compare_digest(token, launch_token):
            raise HTTPException(401, "missing or invalid QGrip launch token")

        async def events() -> AsyncIterator[bytes]:
            """Yield immediate and condition-triggered snapshots until disconnect."""
            # Push the current status immediately, then block until the
            # coordinator signals a change instead of polling on a timer. The
            # bounded wait lets us periodically notice client disconnects.
            #
            # Two named channels share this one authenticated connection:
            #   * ``status`` — the authoritative snapshot the UI must always apply.
            #   * ``notification`` — discrete, ignorable toasts on terminal
            #     transitions; missing one is acceptable (see ARCHITECTURE).
            status, version = owner.status_snapshot()
            last = _status_payload(status)
            last_state = status.state if status else JobState.IDLE
            yield f"event: status\ndata: {last}\n\n".encode()
            while not await request.is_disconnected():
                status, version = await asyncio.to_thread(owner.status_since, version, 1.0)
                payload = _status_payload(status)
                if payload != last:
                    last = payload
                    yield f"event: status\ndata: {payload}\n\n".encode()
                state = status.state if status else JobState.IDLE
                if state != last_state:
                    note = notification_for(status) if status else None
                    if note is not None:
                        frame = json.dumps(note, default=str)
                        yield f"event: notification\ndata: {frame}\n\n".encode()
                last_state = state

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/v1/export/start", dependencies=protected)
    def start_export(body: ExportWire) -> dict[str, object]:
        """Start exclusive capture-to-Parquet projection."""
        return asdict(owner.start_export(Path(body.capture).resolve(), current))

    @app.post("/api/v1/training/start", dependencies=protected)
    def start_training(body: TrainingWire) -> dict[str, object]:
        """Start exclusive training from validated paths and profile configuration."""
        request = TrainingRequest(
            body.subject,
            current,
            tuple(Path(item).resolve() for item in body.inputs),
            body.model or current.model.name,
            not body.discrete,
        )
        return asdict(owner.start_training(request))

    @app.post("/api/v1/training/cancel", dependencies=protected)
    @app.post("/api/v1/inference/stop", dependencies=protected)
    def cancel() -> dict[str, bool]:
        """Request cooperative cancellation of the currently owned workflow."""
        owner.cancel()
        return {"cancelled": True}

    @app.post("/api/v1/inference/start", dependencies=protected)
    def start_inference(body: InferenceWire) -> dict[str, object]:
        """Start exclusive live inference for the selected artifact."""
        return asdict(owner.start_inference(Path(body.model).resolve(), current))

    @app.get("/api/v1/handi/status", dependencies=protected)
    async def handi_status() -> dict[str, object]:
        """Proxy Handi state without assuming ownership of its process or hardware."""
        if not current.dashboard.handi_url:
            return {"configured": False}
        async with httpx.AsyncClient(timeout=current.dashboard.handi_timeout_seconds) as client:
            response = await client.get(f"{current.dashboard.handi_url.rstrip('/')}/api/v1/status")
            response.raise_for_status()
            return {"configured": True, "remote": response.json()}

    @app.post("/api/v1/handi/calibration", dependencies=protected)
    async def handi_calibration(body: CalibrationWire) -> dict[str, object]:
        """Proxy a bounded calibration jog to independently running Handi."""
        if not current.dashboard.handi_url:
            return {"configured": False}
        async with httpx.AsyncClient(timeout=current.dashboard.handi_timeout_seconds) as client:
            response = await client.post(
                f"{current.dashboard.handi_url.rstrip('/')}/api/v1/calibration/jog",
                json=body.model_dump(),
            )
            response.raise_for_status()
            return response.json()

    @app.post("/api/v1/server/stop", dependencies=protected)
    def server_stop() -> dict[str, bool]:
        """Close the coordinator and request dashboard-owned workflow shutdown."""
        owner.close()
        return {"stopping": True}

    assets = Path(__file__).parent / "dashboard"
    if assets.exists():
        app.mount(
            "/assets", StaticFiles(directory=assets / "assets", check_dir=False), name="assets"
        )
        app.mount(
            "/stimuli", StaticFiles(directory=current.assets_root, check_dir=False), name="stimuli"
        )

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            """Serve approved dashboard assets or fall back to the SPA entry document."""
            candidate = (assets / path).resolve()
            if path and candidate.is_file() and assets.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(assets / "index.html")

    return app
