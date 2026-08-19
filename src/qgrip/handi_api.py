"""Optional observer/calibration API around standalone Handi."""

from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from qgrip.handi import HandiRuntime


class JogWire(BaseModel):
    model_config = ConfigDict(extra="forbid")
    joint: str
    delta: float


def create_handi_app(runtime: HandiRuntime) -> FastAPI:
    app = FastAPI(title="QGrip Handi API", version="1.0.0")

    @app.get("/healthz")
    def health() -> dict[str, object]:
        return asdict(runtime.health())

    @app.get("/api/v1/status")
    def status() -> dict[str, object]:
        return asdict(runtime.controller.state)

    @app.post("/api/v1/start")
    def start() -> dict[str, bool]:
        return {"running": runtime.controller.state.running}

    @app.post("/api/v1/stop")
    def stop() -> dict[str, bool]:
        runtime.stop()
        return {"stopping": True}

    @app.post("/api/v1/calibration/jog")
    def jog(body: JogWire) -> dict[str, object]:
        return {"joint": body.joint, "position": runtime.controller.jog(body.joint, body.delta)}

    @app.get("/api/v1/calibration/read")
    def read() -> dict[str, object]:
        return {"positions": dict(runtime.controller.state.positions)}

    @app.post("/api/v1/calibration/save")
    def save() -> dict[str, object]:
        return {"positions": dict(runtime.controller.state.positions), "saved": False}

    return app
