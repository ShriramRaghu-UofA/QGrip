"""Optional observer/calibration API around standalone Handi."""

from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from qgrip.handi import HandiRuntime


class JogWire(BaseModel):
    """Strict HTTP request body for a bounded Handi calibration jog."""

    model_config = ConfigDict(extra="forbid")
    joint: str
    delta: float


def create_handi_app(runtime: HandiRuntime) -> FastAPI:
    """Create the loopback observer and non-persisting calibration API for ``runtime``."""
    app = FastAPI(title="QGrip Handi API", version="1.0.0")

    @app.get("/healthz")
    def health() -> dict[str, object]:
        """Expose aggregate readiness for simple process monitoring."""
        return asdict(runtime.health())

    @app.get("/api/v1/status")
    def status() -> dict[str, object]:
        """Return the controller's latest immutable observable state."""
        return asdict(runtime.controller.state)

    @app.post("/api/v1/start")
    def start() -> dict[str, bool]:
        """Report whether the independently owned runtime is currently running."""
        return {"running": runtime.controller.state.running}

    @app.post("/api/v1/stop")
    def stop() -> dict[str, bool]:
        """Request cooperative runtime shutdown; hardware ownership remains external."""
        runtime.stop()
        return {"stopping": True}

    @app.post("/api/v1/calibration/jog")
    def jog(body: JogWire) -> dict[str, object]:
        """Apply one controller-clamped, step-bounded calibration movement."""
        return {"joint": body.joint, "position": runtime.controller.jog(body.joint, body.delta)}

    @app.get("/api/v1/calibration/read")
    def read() -> dict[str, object]:
        """Read current joint positions without modifying the profile."""
        return {"positions": dict(runtime.controller.state.positions)}

    @app.post("/api/v1/calibration/save")
    def save() -> dict[str, object]:
        """Explicitly report that calibration observations are not persisted here."""
        return {"positions": dict(runtime.controller.state.positions), "saved": False}

    return app
