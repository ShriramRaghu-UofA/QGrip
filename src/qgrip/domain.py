"""Immutable domain values. Framework-specific wire models do not belong here."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

DeviceKind = Literal["sifi", "myo_ble", "myo_dongle", "synthetic"]
ModelName = Literal["transformer", "cnn1d", "cnn2d", "dense"]
InferenceBackend = Literal["auto", "torch", "onnx"]
JobState = Literal["idle", "running", "completed", "cancelled", "failed"]


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    kind: DeviceKind = "synthetic"
    sample_rate_hz: float = 200.0
    channels: int = 8
    address: str | None = None
    port: str | None = None
    seed: int = 7


@dataclass(frozen=True, slots=True)
class SGTConfig:
    gestures: tuple[str, ...] = (
        "rest",
        "close",
        "open",
        "wrist_flexion",
        "wrist_extension",
        "pronation",
        "supination",
    )
    trials: int = 3
    duration_seconds: float = 4.0
    practice: bool = True
    proportional: bool = True
    activation_calibration: bool = True


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: ModelName = "transformer"
    preset_version: int = 1


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    backend: InferenceBackend = "auto"
    confidence_gate: float = 0.6
    interval_seconds: float = 0.05
    switch_samples: int = 3


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    handi_url: str | None = None


@dataclass(frozen=True, slots=True)
class JointLimit:
    name: str
    minimum: float
    maximum: float
    start: float

    def clamp(self, value: float) -> float:
        return min(self.maximum, max(self.minimum, value))


@dataclass(frozen=True, slots=True)
class GripPreset:
    name: str
    positions: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class HandiConfig:
    enabled: bool = False
    rpc_socket: str = "/var/run/arduino-router.sock"
    rpc_host: str = "127.0.0.1"
    rpc_port: int = 5000
    rpc_timeout_seconds: float = 1.0
    api_enabled: bool = False
    api_host: str = "127.0.0.1"
    api_port: int = 8770
    step: float = 5.0
    map_flexion_to_grips: bool = False
    joints: tuple[JointLimit, ...] = ()
    grips: tuple[GripPreset, ...] = ()
    gesture_mapping: tuple[tuple[str, str], ...] = (
        ("open", "open"),
        ("close", "close"),
    )


@dataclass(frozen=True, slots=True)
class QGripProfile:
    schema_version: int
    path: Path
    data_root: Path
    assets_root: Path
    device: DeviceConfig
    sgt: SGTConfig
    model: ModelConfig
    inference: InferenceConfig
    dashboard: DashboardConfig
    handi: HandiConfig | None = None


@dataclass(frozen=True, slots=True)
class SGTRequest:
    subject: str
    profile: QGripProfile
    proportional: bool


@dataclass(frozen=True, slots=True)
class SGTProgress:
    state: JobState
    gesture: str | None = None
    trial: int = 0
    total_trials: int = 0
    elapsed_seconds: float = 0.0
    activation: float = 0.0
    capture: Path | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    path: Path
    kind: str
    subject: str
    created_at: str
    device: DeviceKind
    sample_rate_hz: float
    channels: int
    classes: tuple[str, ...]
    proportional: bool
    complete: bool = True


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    subject: str
    profile: QGripProfile
    inputs: tuple[Path, ...]
    model: ModelName
    proportional: bool


@dataclass(frozen=True, slots=True)
class EpochMetric:
    epoch: int
    loss: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class Prediction:
    gesture: str
    confidence: float
    activation: float
    latency_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class ControllerState:
    running: bool = False
    healthy: bool = True
    positions: tuple[tuple[str, float], ...] = ()
    grip: str | None = None
    prediction: Prediction | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class Health:
    ok: bool
    device: bool
    model: bool
    rpc: bool
    message: str = ""


@dataclass(frozen=True, slots=True)
class JobStatus:
    id: str
    kind: str
    state: JobState
    progress: float = 0.0
    message: str = ""
    result: str | None = None
    metrics: tuple[EpochMetric, ...] = field(default_factory=tuple)
    prediction: Prediction | None = None
