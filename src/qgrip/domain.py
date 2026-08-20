"""Immutable domain values. Framework-specific wire models do not belong here."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


def activation_target(gesture: str, fraction: float) -> float:
    """Triangle-ramp activation prompt over a normalized presentation fraction.

    The proportional SGT "metronome" prompts the operator to ramp effort up and
    then back down within a single hold: ``rest`` stays relaxed, while every
    other gesture ramps linearly from ``0`` at the start up to a full
    contraction at the midpoint and back to ``0`` at the end.  ``fraction`` is
    the elapsed position in ``[0, 1]``.  The same helper drives the on-screen
    prompt during capture and reconstructs the per-sample proxy label during
    export, so the labels always match what the operator was asked to follow.
    """
    if gesture == "rest":
        return 0.0
    clamped = min(1.0, max(0.0, fraction))
    return 1.0 - abs(2.0 * clamped - 1.0)


class DeviceKind(StrEnum):
    SIFI = "sifi"
    MYO_BLE = "myo_ble"
    MYO_DONGLE = "myo_dongle"
    SYNTHETIC = "synthetic"


class ModelName(StrEnum):
    TRANSFORMER = "transformer"
    CNN1D = "cnn1d"
    CNN2D = "cnn2d"
    DENSE = "dense"


class InferenceBackend(StrEnum):
    AUTO = "auto"
    TORCH = "torch"
    ONNX = "onnx"


class NormalizationMode(StrEnum):
    WINDOW_ZSCORE = "window_zscore"
    SIGNED_8BIT = "signed_8bit"
    DATASET_STANDARDIZE = "dataset_standardize"


class JobState(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class SGTCommand(StrEnum):
    ABORT = "abort"
    PAUSE = "pause"
    RESUME = "resume"
    REPEAT = "repeat"


class SignalHealthSeverity(StrEnum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"
    WARMING_UP = "warming_up"


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    kind: DeviceKind = DeviceKind.SYNTHETIC
    sample_rate_hz: float = 200.0
    channels: int = 8
    address: str | None = None
    port: str | None = None
    seed: int = 7


@dataclass(frozen=True, slots=True)
class HealthConfig:
    window_seconds: float = 5.0
    stale_after_seconds: float | None = 2.0
    minimum_rate_ratio: float | None = 0.9
    maximum_rate_ratio: float | None = 1.1
    maximum_missing_fraction: float | None = 0.0
    maximum_lost_samples: int | None = 0


@dataclass(frozen=True, slots=True)
class AcquisitionConfig:
    ring_buffer_seconds: float = 10.0
    ack_timeout_seconds: float = 2.0
    capture_log_enabled: bool = True
    capture_frame_target_bytes: int = 1 << 20
    capture_flush_interval_seconds: float = 1.0
    capture_compression_level: int | None = None
    capture_fsync_on_boundary: bool = True
    health: HealthConfig = field(default_factory=HealthConfig)


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
    preparation_seconds: float = 3.0
    practice: bool = True
    proportional: bool = True
    progress_interval_seconds: float = 0.05


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: ModelName = ModelName.TRANSFORMER
    architecture: tuple[tuple[str, float | int], ...] = ()


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-4
    validation_fraction: float = 0.2
    training_window_seconds: float = 1.0
    dataset_stride_seconds: float = 0.005
    stft_n_fft: int | None = None
    stft_hop_samples: int | None = None
    activation_energy_window_seconds: float = 0.1
    activation_reference_quantile: float = 0.9
    activation_smoothing_threshold: float = 0.25
    activation_loss_weight: float = 1.0
    weight_decay: float = 1e-4
    normalization: NormalizationMode = NormalizationMode.DATASET_STANDARDIZE
    seed: int = 42
    export_onnx: bool = True


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    backend: InferenceBackend = InferenceBackend.AUTO
    confidence_gate: float = 0.6
    inference_period_seconds: float = 1 / 60
    switch_predictions: int = 3
    idle_poll_seconds: float = 0.002
    maximum_wait_seconds: float = 0.01


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    handi_url: str | None = None
    handi_timeout_seconds: float = 2.0


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
    rpc_timeout_seconds: float = 1.0
    api_enabled: bool = False
    api_host: str = "127.0.0.1"
    api_port: int = 8770
    step: float = 5.0
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
    acquisition: AcquisitionConfig
    sgt: SGTConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig
    dashboard: DashboardConfig
    handi: HandiConfig | None = None


@dataclass(frozen=True, slots=True)
class SGTRequest:
    subject: str
    profile: QGripProfile
    proportional: bool
    auto: bool = True


@dataclass(frozen=True, slots=True)
class SGTProgress:
    state: JobState
    gesture: str | None = None
    stage: str | None = None
    instruction: str | None = None
    stimulus_image: str | None = None
    trial: int = 0
    total_trials: int = 0
    elapsed_seconds: float = 0.0
    duration_seconds: float = 0.0
    activation: float = 0.0
    capture: Path | None = None
    error: str | None = None
    awaiting_command: bool = False


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
    training_loss: float = 0.0
    training_accuracy: float = 0.0


@dataclass(frozen=True, slots=True)
class ClassSampleCount:
    """Per-class window counts for the training and validation splits."""

    label: str
    training: int
    validation: int

    @property
    def total(self) -> int:
        return self.training + self.validation


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    """Static dataset shape surfaced once training starts building windows."""

    training_samples: int
    validation_samples: int
    window_size: int
    classes: tuple[ClassSampleCount, ...] = ()

    @property
    def total_samples(self) -> int:
        return self.training_samples + self.validation_samples


@dataclass(frozen=True, slots=True)
class Prediction:
    gesture: str
    confidence: float
    activation: float
    latency_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class LiveSignalHealth:
    """QGrip wire-safe summary of streamer device and consumer health."""

    severity: SignalHealthSeverity = SignalHealthSeverity.WARMING_UP
    warnings: tuple[str, ...] = ()
    missing_values: int = 0
    lost_samples: int = 0
    malformed_packets: int = 0
    misaligned_packets: int = 0
    consumer_overruns: int = 0


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
    gesture: str | None = None
    stage: str | None = None
    instruction: str | None = None
    stimulus_image: str | None = None
    elapsed_seconds: float = 0.0
    duration_seconds: float = 0.0
    activation: float = 0.0
    result: str | None = None
    metrics: tuple[EpochMetric, ...] = field(default_factory=tuple)
    training_summary: TrainingSummary | None = None
    prediction: Prediction | None = None
    health: LiveSignalHealth | None = None
    awaiting_command: bool = False
