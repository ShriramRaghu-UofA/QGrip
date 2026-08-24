"""Immutable domain values. Framework-specific wire models do not belong here."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

DEFAULT_ACTIVATION_LEVELS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)


def resolve_stft_dimensions(
    sample_rate_hz: float,
    model_window_seconds: float,
    stft_window_seconds: float,
    stft_hop_seconds: float,
) -> tuple[int, int, int]:
    """Resolve time-based STFT settings to an exactly aligned sample geometry."""
    if sample_rate_hz <= 0:
        raise ValueError("sample rate must be positive")
    if model_window_seconds <= 0 or stft_window_seconds <= 0 or stft_hop_seconds <= 0:
        raise ValueError("model and STFT timing values must be positive")
    if stft_window_seconds > model_window_seconds:
        raise ValueError("STFT analysis window must be no longer than the model input window")
    window_size = max(8, round(sample_rate_hz * model_window_seconds))
    n_fft = math.floor(sample_rate_hz * stft_window_seconds)
    if n_fft < 4:
        raise ValueError("STFT analysis window must contain at least four samples")
    if n_fft > window_size:
        raise ValueError("STFT analysis window must fit within the model input window")
    hop_length = max(1, round(sample_rate_hz * stft_hop_seconds))
    if hop_length > n_fft:
        raise ValueError("STFT hop must be no larger than its analysis window")
    if (window_size - n_fft) % hop_length:
        raise ValueError(
            "STFT window and hop do not align with the model input window; "
            "choose compatible training timing values"
        )
    return window_size, n_fft, hop_length


#: UNO Q LED matrix geometry (8 rows x 13 cols) — see GripPreset.led_frame.
LED_MATRIX_ROWS = 8
LED_MATRIX_COLS = 13
LED_MATRIX_PIXELS = LED_MATRIX_ROWS * LED_MATRIX_COLS


class DeviceKind(StrEnum):
    """Acquisition transports supported by a profile's :class:`DeviceConfig`."""

    SIFI = "sifi"
    MYO_BLE = "myo_ble"
    MYO_DONGLE = "myo_dongle"
    SYNTHETIC = "synthetic"


class ModelName(StrEnum):
    """Model architectures that QGrip can construct and load from checkpoints."""

    TRANSFORMER = "transformer"
    CNN1D = "cnn1d"
    CNN2D = "cnn2d"
    DENSE = "dense"


class InferenceBackend(StrEnum):
    """Runtime selection policy for a trained model artifact."""

    AUTO = "auto"
    TORCH = "torch"
    ONNX = "onnx"


class NormalizationMode(StrEnum):
    """Raw-EMG scaling strategy embedded in the trained model graph."""

    WINDOW_ZSCORE = "window_zscore"
    SIGNED_8BIT = "signed_8bit"
    DATASET_STANDARDIZE = "dataset_standardize"


class JobState(StrEnum):
    """Lifecycle state reported by :class:`~qgrip.runtime.workflows.WorkflowCoordinator`."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class SGTCommand(StrEnum):
    """Operator commands accepted while a manual SGT capture is awaiting input."""

    ABORT = "abort"
    PAUSE = "pause"
    RESUME = "resume"
    REPEAT = "repeat"


class SignalHealthSeverity(StrEnum):
    """Ordered health assessment for a live acquisition stream."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"
    WARMING_UP = "warming_up"


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    """Physical or synthetic EMG source selected by a profile.

    ``kind`` chooses the transport. ``sample_rate_hz`` and ``channels`` are the
    nominal stream identity and must agree with datasets and checkpoints. For
    ``kind == DeviceKind.SIFI``, ``sample_rate_hz`` selects the bridge's onboard
    EMG sample rate and ``imu_sample_rate_hz`` selects its onboard IMU sample
    rate, both validated against sifi-streamer's own supported values. For
    ``{DeviceKind.MYO_BLE, DeviceKind.MYO_DONGLE}``, ``sample_rate_hz`` is fixed
    at 200Hz (not configurable) and ``imu_sample_rate_hz`` must stay ``None``
    since Myo has no configurable/consumed IMU rate concept.
    ``address`` is used by SiFi and Myo BLE; ``port`` is used by SiFi and the
    Myo dongle. ``seed`` makes synthetic acquisition reproducible.
    """

    kind: DeviceKind = DeviceKind.SYNTHETIC
    sample_rate_hz: float = 200.0
    channels: int = 8
    address: str | None = None
    port: str | None = None
    seed: int = 7
    imu_sample_rate_hz: float | None = None


@dataclass(frozen=True, slots=True)
class HealthConfig:
    """Thresholds used to judge acquisition continuity and quality.

    The health window aggregates recent packets. Optional rate, stale, missing,
    and loss limits may be set to ``None`` to disable that individual check.
    Rate ratios are measured against ``DeviceConfig.sample_rate_hz``.
    """

    window_seconds: float = 5.0
    stale_after_seconds: float | None = 2.0
    minimum_rate_ratio: float | None = 0.9
    maximum_rate_ratio: float | None = 1.1
    maximum_missing_fraction: float | None = 0.0
    maximum_lost_samples: int | None = 0


@dataclass(frozen=True, slots=True)
class AcquisitionConfig:
    """Streamer buffering, durable capture-log, and health-monitor settings.

    ``ring_buffer_seconds`` bounds live consumer history. Capture options control
    the authoritative compressed log's frame size, flushing, compression, and
    boundary fsync behavior. ``ack_timeout_seconds`` bounds device startup.
    """

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
    """Screen-guided-training presentation plan.

    ``gestures`` is an ordered, unique class list containing ``rest``. Each
    gesture is presented once per ``trials`` pass. ``duration_seconds`` is each
    recorded presentation; preparation and optional practice are unlabelled.
    In proportional mode non-rest prompts use calibrated, held activation levels.
    """

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
    activation_levels: tuple[float, ...] = DEFAULT_ACTIVATION_LEVELS
    activation_tolerance: float = 0.1
    activation_smoothing_seconds: float = 0.1
    calibration_rest_seconds: float = 4.0
    calibration_max_seconds: float = 4.0


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Chosen classifier and its architecture-specific, validated parameters.

    ``architecture`` is stored as an immutable ordered key/value tuple so
    profiles and checkpoints have deterministic serialization. Valid keys depend
    on ``name`` and are enforced by the profile parser.
    """

    name: ModelName = ModelName.TRANSFORMER
    architecture: tuple[tuple[str, float | int], ...] = ()


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Windowing, optimization, activation-target, and export settings.

    Training uses windows of ``training_window_seconds`` separated by
    ``dataset_stride_seconds``. STFT analysis-window and hop durations resolve
    to samples at the configured device rate. The activation fields calibrate
    proportional targets from causal EMG energy, while ``normalization`` becomes
    model state.
    """

    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-4
    validation_fraction: float = 0.2
    training_window_seconds: float = 1.0
    dataset_stride_seconds: float = 0.005
    stft_window_seconds: float = 0.1
    stft_hop_seconds: float = 0.02
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
    """Live inference backend, cadence, confidence gate, and switch debounce.

    Predictions below ``confidence_gate`` are treated as ``rest`` by live
    adapters. A changed gesture must persist for ``switch_predictions`` model
    outputs before it is accepted. Poll/wait values control stream consumption.
    """

    backend: InferenceBackend = InferenceBackend.AUTO
    confidence_gate: float = 0.6
    inference_period_seconds: float = 1 / 60
    switch_predictions: int = 3
    idle_poll_seconds: float = 0.002
    maximum_wait_seconds: float = 0.01


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    """Local dashboard bind address."""

    host: str = "127.0.0.1"
    port: int = 8765


@dataclass(frozen=True, slots=True)
class JointLimit:
    """Verified safe range for one joint.

    ``minimum`` doubles as both this joint's startup position (sent by
    ``HandController.apply_start_pose``) and its "fully open" reference that
    open/close gestures blend away from (``HandController.apply_prediction``).
    """

    name: str
    minimum: float
    maximum: float

    def clamp(self, value: float) -> float:
        """Return ``value`` limited to this joint's inclusive safe range."""
        return min(self.maximum, max(self.minimum, value))


@dataclass(frozen=True, slots=True)
class GripPreset:
    """Named multi-joint target pose used by a Handi gesture mapping.

    ``led_frame``, when set, is a 104-value grayscale (0-7) LED matrix frame
    (row-major, 8 rows x 13 cols) sent alongside the pose via the Router's
    ``set_led_frame`` RPC so the matrix reflects the active grip.
    """

    name: str
    positions: tuple[tuple[str, float], ...]
    led_frame: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class HandiConfig:
    """Standalone Handi RPC, safety, and gesture-to-movement configuration.

    Enabled control requires at least one verified ``JointLimit``. Every motion
    is clamped to those limits. ``step`` is the calibration jog's maximum raw-unit
    delta per call (see ``HandController.jog``) — unrelated to gesture-driven
    motion. ``openness_step`` is the maximum full-activation fraction (0..1) of
    full open<->closed travel that one mapped ``open``/``close`` prediction may
    advance the active grip's blend by; named mappings select a preset.
    """

    enabled: bool = False
    rpc_socket: str = "/var/run/arduino-router.sock"
    rpc_timeout_seconds: float = 1.0
    step: float = 5.0
    openness_step: float = 0.1
    joints: tuple[JointLimit, ...] = ()
    grips: tuple[GripPreset, ...] = ()
    gesture_mapping: tuple[tuple[str, str], ...] = (
        ("open", "open"),
        ("close", "close"),
    )


@dataclass(frozen=True, slots=True)
class QGripProfile:
    """Fully resolved immutable profile consumed by every QGrip adapter.

    Paths are absolute after loading, even when their JSON representation was
    relative to the profile file. Construct profiles through ``load_profile`` so
    the schema, cross-field constraints, and relative-path semantics are kept.
    """

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
    """Typed request to run screen-guided capture for one validated subject."""

    subject: str
    profile: QGripProfile
    proportional: bool
    auto: bool = True


@dataclass(frozen=True, slots=True)
class SGTProgress:
    """Incremental SGT state used by CLI, HTTP, and UI progress adapters."""

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
    measured_activation: float = 0.0
    in_tolerance: bool = False
    capture: Path | None = None
    error: str | None = None
    awaiting_command: bool = False


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    """Validated identity and completeness metadata for a capture or Parquet artifact."""

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
    """Typed request selecting training inputs, architecture, and output mode."""

    subject: str
    profile: QGripProfile
    inputs: tuple[Path, ...]
    model: ModelName
    proportional: bool


@dataclass(frozen=True, slots=True)
class EpochMetric:
    """Loss and accuracy values recorded after one training epoch."""

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
        """Return all windows assigned to this class across both split partitions."""
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
        """Return the total number of constructed windows in both split partitions."""
        return self.training_samples + self.validation_samples


@dataclass(frozen=True, slots=True)
class Prediction:
    """One classifier output: winning gesture, confidence, and proportional effort."""

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
    """Observable state of the independently owned Handi runtime and controller."""

    running: bool = False
    healthy: bool = True
    positions: tuple[tuple[str, float], ...] = ()
    grip: str | None = None
    openness: float = 1.0
    prediction: Prediction | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class Health:
    """Readiness result for the device, model, and Handi RPC dependencies."""

    ok: bool
    device: bool
    model: bool
    rpc: bool
    message: str = ""


@dataclass(frozen=True, slots=True)
class JobStatus:
    """Authoritative coordinator snapshot for the single process-local workflow job."""

    id: str
    kind: str
    state: JobState
    progress: float = 0.0
    message: str = ""
    gesture: str | None = None
    trial: int = 0
    stage: str | None = None
    instruction: str | None = None
    stimulus_image: str | None = None
    elapsed_seconds: float = 0.0
    duration_seconds: float = 0.0
    activation: float = 0.0
    measured_activation: float = 0.0
    in_tolerance: bool = False
    result: str | None = None
    metrics: tuple[EpochMetric, ...] = field(default_factory=tuple)
    training_summary: TrainingSummary | None = None
    prediction: Prediction | None = None
    health: LiveSignalHealth | None = None
    awaiting_command: bool = False
