"""Strict versioned profile parsing, defaults, and atomic persistence.

Profiles are JSON documents at the application's configuration boundary.  They
are deliberately closed schemas: unknown keys, unsupported versions, malformed
values, and invalid cross-field combinations fail during loading.  Relative
``data_root`` and ``assets_root`` paths are resolved beside the profile rather
than against the caller's current directory.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from sifi_streamer.sifi.sensor_profile import EmgConfiguration, ImuConfiguration

from qgrip.core.domain import (
    DEFAULT_ACTIVATION_LEVELS,
    LED_MATRIX_PIXELS,
    AcquisitionConfig,
    DashboardConfig,
    DeviceConfig,
    DeviceKind,
    GripPreset,
    HandiConfig,
    HealthConfig,
    InferenceBackend,
    InferenceConfig,
    JointLimit,
    ModelConfig,
    ModelName,
    NormalizationMode,
    QGripProfile,
    SGTConfig,
    TrainingConfig,
    resolve_stft_dimensions,
)
from qgrip.core.errors import ValidationError

SCHEMA_VERSION = 1


def _object(value: object, name: str) -> dict[str, object]:
    """Require a JSON object and retain its boundary-local object type."""
    if not isinstance(value, dict):
        raise ValidationError(f"{name} must be an object")
    return cast(dict[str, object], value)


def _only(data: dict[str, object], allowed: set[str], name: str) -> None:
    """Reject keys outside the closed schema for a named JSON object."""
    unknown = set(data) - allowed
    if unknown:
        raise ValidationError(f"unknown {name} keys: {', '.join(sorted(unknown))}")


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    """Parse a finite JSON number, optionally requiring it to be strictly positive."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValidationError(f"{name} must be finite{' and positive' if positive else ''}")
    return result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    """Parse a non-boolean JSON integer that meets an inclusive lower bound."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValidationError(f"{name} must be an integer >= {minimum}")
    return value


def _optional_integer(value: object, name: str, *, minimum: int) -> int | None:
    """Parse an integer constraint or preserve JSON ``null`` as ``None``."""
    return None if value is None else _integer(value, name, minimum=minimum)


def _optional_finite(value: object, name: str, *, positive: bool = False) -> float | None:
    """Parse a finite-number constraint or preserve JSON ``null`` as ``None``."""
    return None if value is None else _finite(value, name, positive=positive)


def _led_frame(value: object, name: str) -> tuple[int, ...] | None:
    """Parse an optional 104-value grayscale (0-7) LED matrix frame, or ``None``."""
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != LED_MATRIX_PIXELS:
        raise ValidationError(f"{name} must be an array of {LED_MATRIX_PIXELS} values")
    frame = tuple(_integer(pixel, f"{name}[]", minimum=0) for pixel in value)
    if any(pixel > 7 for pixel in frame):
        raise ValidationError(f"{name} values must be in 0..7")
    return frame


def _boolean(value: object, name: str) -> bool:
    """Require a JSON boolean without accepting integer lookalikes."""
    if not isinstance(value, bool):
        raise ValidationError(f"{name} must be a boolean")
    return value


def _string(value: object, name: str) -> str:
    """Require and trim a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _enum[EnumValue: StrEnum](value: object, enum: type[EnumValue], name: str) -> EnumValue:
    """Parse a string into a supported string-enum member with a clear error."""
    text = _string(value, name)
    try:
        return enum(text)
    except ValueError as exc:
        raise ValidationError(f"unsupported {name}: {text}") from exc


def _optional_string(value: object, name: str) -> str | None:
    """Parse a non-empty string or preserve JSON ``null`` as ``None``."""
    return None if value is None else _string(value, name)


def _resolve(base: Path, value: object, name: str) -> Path:
    """Resolve a configured filesystem path relative to its profile directory."""
    path = Path(_string(value, name)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _parse_device(raw: object) -> DeviceConfig:
    """Validate the ``device`` section and its transport-specific fields."""
    data = _object(raw, "device")
    _only(
        data,
        {"kind", "sample_rate_hz", "channels", "address", "port", "seed", "imu_sample_rate_hz"},
        "device",
    )
    kind = _enum(data.get("kind", DeviceKind.SYNTHETIC), DeviceKind, "device.kind")
    address = _optional_string(data.get("address"), "device.address")
    port = _optional_string(data.get("port"), "device.port")
    if kind == "myo_ble" and port is not None:
        raise ValidationError("myo_ble cannot define device.port")
    if kind == "myo_dongle" and address is not None:
        raise ValidationError("myo_dongle cannot define device.address")
    sample_rate_hz = _finite(
        data.get("sample_rate_hz", 200), "device.sample_rate_hz", positive=True
    )
    if kind in {DeviceKind.MYO_BLE, DeviceKind.MYO_DONGLE} and sample_rate_hz != 200:
        raise ValidationError(f"{kind} has a fixed device.sample_rate_hz of 200")
    raw_imu = data.get("imu_sample_rate_hz")
    if raw_imu is not None and kind != DeviceKind.SIFI:
        raise ValidationError(
            f"device.imu_sample_rate_hz is not applicable for device.kind = {kind}"
        )
    imu_sample_rate_hz: float | None = None
    if kind == DeviceKind.SIFI:
        resolved_imu_rate = (
            _finite(raw_imu, "device.imu_sample_rate_hz", positive=True)
            if raw_imu is not None
            else ImuConfiguration().sample_rate_hz
        )
        try:
            EmgConfiguration(sample_rate_hz=round(sample_rate_hz))
            ImuConfiguration(sample_rate_hz=round(resolved_imu_rate))
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        imu_sample_rate_hz = resolved_imu_rate
    return DeviceConfig(
        kind=kind,
        sample_rate_hz=sample_rate_hz,
        channels=_integer(data.get("channels", 8), "device.channels", minimum=1),
        address=address,
        port=port,
        seed=_integer(data.get("seed", 7), "device.seed"),
        imu_sample_rate_hz=imu_sample_rate_hz,
    )


def _parse_acquisition(raw: object) -> AcquisitionConfig:
    """Validate streamer buffering, capture-log durability, and health settings."""
    data = _object(raw, "acquisition")
    _only(
        data,
        {
            "ring_buffer_seconds",
            "ack_timeout_seconds",
            "capture_log_enabled",
            "capture_frame_target_bytes",
            "capture_flush_interval_seconds",
            "capture_compression_level",
            "capture_fsync_on_boundary",
            "health",
        },
        "acquisition",
    )
    health_data = _object(data.get("health", {}), "acquisition.health")
    _only(
        health_data,
        {
            "window_seconds",
            "stale_after_seconds",
            "minimum_rate_ratio",
            "maximum_rate_ratio",
            "maximum_missing_fraction",
            "maximum_lost_samples",
        },
        "acquisition.health",
    )
    health = HealthConfig(
        window_seconds=_finite(
            health_data.get("window_seconds", 5), "acquisition.health.window_seconds", positive=True
        ),
        stale_after_seconds=_optional_finite(
            health_data.get("stale_after_seconds", 2),
            "acquisition.health.stale_after_seconds",
            positive=True,
        ),
        minimum_rate_ratio=_optional_finite(
            health_data.get("minimum_rate_ratio", 0.9), "acquisition.health.minimum_rate_ratio"
        ),
        maximum_rate_ratio=_optional_finite(
            health_data.get("maximum_rate_ratio", 1.1), "acquisition.health.maximum_rate_ratio"
        ),
        maximum_missing_fraction=_optional_finite(
            health_data.get("maximum_missing_fraction", 0),
            "acquisition.health.maximum_missing_fraction",
        ),
        maximum_lost_samples=_optional_integer(
            health_data.get("maximum_lost_samples", 0),
            "acquisition.health.maximum_lost_samples",
            minimum=0,
        ),
    )
    return AcquisitionConfig(
        ring_buffer_seconds=_finite(
            data.get("ring_buffer_seconds", 10), "acquisition.ring_buffer_seconds", positive=True
        ),
        ack_timeout_seconds=_finite(
            data.get("ack_timeout_seconds", 2), "acquisition.ack_timeout_seconds", positive=True
        ),
        capture_log_enabled=_boolean(
            data.get("capture_log_enabled", True), "acquisition.capture_log_enabled"
        ),
        capture_frame_target_bytes=_integer(
            data.get("capture_frame_target_bytes", 1 << 20),
            "acquisition.capture_frame_target_bytes",
            minimum=1,
        ),
        capture_flush_interval_seconds=_finite(
            data.get("capture_flush_interval_seconds", 1),
            "acquisition.capture_flush_interval_seconds",
            positive=True,
        ),
        capture_compression_level=_optional_integer(
            data.get("capture_compression_level"),
            "acquisition.capture_compression_level",
            minimum=-22,
        ),
        capture_fsync_on_boundary=_boolean(
            data.get("capture_fsync_on_boundary", True), "acquisition.capture_fsync_on_boundary"
        ),
        health=health,
    )


def _parse_sgt(raw: object) -> SGTConfig:
    """Validate SGT class order, timing, operator-practice, and target mode."""
    data = _object(raw, "sgt")
    _only(
        data,
        {
            "gestures",
            "trials",
            "duration_seconds",
            "preparation_seconds",
            "practice",
            "proportional",
            "progress_interval_seconds",
            "activation_levels",
            "activation_tolerance",
            "activation_smoothing_seconds",
            "calibration_rest_seconds",
            "calibration_max_seconds",
        },
        "sgt",
    )
    gestures_raw = data.get(
        "gestures",
        [
            "rest",
            "close",
            "open",
            "wrist_flexion",
            "wrist_extension",
            "pronation",
            "supination",
        ],
    )
    if not isinstance(gestures_raw, list):
        raise ValidationError("sgt.gestures must be a list")
    gestures = tuple(_string(item, "gesture") for item in gestures_raw)
    if len(gestures) < 2 or len(set(gestures)) != len(gestures):
        raise ValidationError("sgt.gestures must contain at least two unique names")
    if "rest" not in gestures:
        raise ValidationError("sgt.gestures must include rest")
    preparation_seconds = _finite(data.get("preparation_seconds", 3), "sgt.preparation_seconds")
    if preparation_seconds < 0:
        raise ValidationError("sgt.preparation_seconds must be non-negative")
    activation_tolerance = _finite(
        data.get("activation_tolerance", 0.1), "sgt.activation_tolerance"
    )
    if activation_tolerance <= 0 or activation_tolerance > 1:
        raise ValidationError("sgt.activation_tolerance must be in (0, 1]")
    return SGTConfig(
        gestures=gestures,
        trials=_integer(data.get("trials", 3), "sgt.trials", minimum=1),
        duration_seconds=_finite(
            data.get("duration_seconds", 4), "sgt.duration_seconds", positive=True
        ),
        preparation_seconds=preparation_seconds,
        practice=_boolean(data.get("practice", True), "sgt.practice"),
        proportional=_boolean(data.get("proportional", True), "sgt.proportional"),
        progress_interval_seconds=_finite(
            data.get("progress_interval_seconds", 0.05),
            "sgt.progress_interval_seconds",
            positive=True,
        ),
        activation_levels=_activation_levels(data.get("activation_levels")),
        activation_tolerance=activation_tolerance,
        activation_smoothing_seconds=_finite(
            data.get("activation_smoothing_seconds", 0.1),
            "sgt.activation_smoothing_seconds",
            positive=True,
        ),
        calibration_rest_seconds=_finite(
            data.get("calibration_rest_seconds", 4), "sgt.calibration_rest_seconds", positive=True
        ),
        calibration_max_seconds=_finite(
            data.get("calibration_max_seconds", 4), "sgt.calibration_max_seconds", positive=True
        ),
    )


def _activation_levels(raw: object) -> tuple[float, ...]:
    """Validate the canonical ascending proportional target levels."""
    values = list(DEFAULT_ACTIVATION_LEVELS) if raw is None else raw
    if not isinstance(values, list) or not values:
        raise ValidationError("sgt.activation_levels must be a non-empty list")
    levels = tuple(_finite(value, "sgt.activation_levels") for value in values)
    if any(level <= 0 or level > 1 for level in levels) or tuple(sorted(set(levels))) != levels:
        raise ValidationError(
            "sgt.activation_levels must be unique and strictly ascending in (0, 1]"
        )
    return levels


def _parse_model(raw: object) -> ModelConfig:
    """Validate the selected classifier and only its supported architecture options."""
    data = _object(raw, "model")
    _only(data, {"name", "architecture"}, "model")
    name = _enum(data.get("name", ModelName.TRANSFORMER), ModelName, "model.name")
    architecture = _object(data.get("architecture", {}), "model.architecture")
    allowed = {
        ModelName.TRANSFORMER: {"d_model", "nhead", "dim_feedforward", "dropout"},
        ModelName.CNN1D: {"hidden_channels", "dropout"},
        ModelName.CNN2D: {"hidden_channels", "dropout"},
        ModelName.DENSE: {"hidden_dim", "dropout"},
    }[name]
    _only(architecture, allowed, "model.architecture")
    parsed_architecture: list[tuple[str, float | int]] = []
    for key, value in architecture.items():
        parsed_architecture.append(
            (
                key,
                _finite(value, f"model.architecture.{key}")
                if key == "dropout"
                else _integer(value, f"model.architecture.{key}", minimum=1),
            )
        )
    return ModelConfig(name, tuple(parsed_architecture))


def _parse_training(raw: object, device: DeviceConfig) -> TrainingConfig:
    """Validate training windows, optimization, activation calibration, and export."""
    data = _object(raw, "training")
    _only(
        data,
        {
            "epochs",
            "batch_size",
            "learning_rate",
            "validation_fraction",
            "training_window_seconds",
            "dataset_stride_seconds",
            "stft_window_seconds",
            "stft_hop_seconds",
            "activation_energy_window_seconds",
            "activation_reference_quantile",
            "activation_smoothing_threshold",
            "activation_loss_weight",
            "weight_decay",
            "normalization",
            "seed",
            "export_onnx",
        },
        "training",
    )
    validation_fraction = _finite(
        data.get("validation_fraction", 0.2), "training.validation_fraction", positive=True
    )
    if validation_fraction >= 1:
        raise ValidationError("training.validation_fraction must be less than 1")
    stft_window_seconds = _finite(
        data.get("stft_window_seconds", 0.1),
        "training.stft_window_seconds",
        positive=True,
    )
    stft_hop_seconds = _finite(
        data.get("stft_hop_seconds", 0.02),
        "training.stft_hop_seconds",
        positive=True,
    )
    if stft_hop_seconds > stft_window_seconds:
        raise ValidationError(
            "training.stft_hop_seconds must be no larger than training.stft_window_seconds"
        )
    activation_reference_quantile = _finite(
        data.get("activation_reference_quantile", 0.9),
        "training.activation_reference_quantile",
        positive=True,
    )
    if activation_reference_quantile >= 1:
        raise ValidationError("training.activation_reference_quantile must be less than 1")
    activation_smoothing_threshold = _finite(
        data.get("activation_smoothing_threshold", 0.25),
        "training.activation_smoothing_threshold",
        positive=True,
    )
    if activation_smoothing_threshold > 1:
        raise ValidationError("training.activation_smoothing_threshold must be at most 1")
    training_window_seconds = _finite(
        data.get("training_window_seconds", 1.0),
        "training.training_window_seconds",
        positive=True,
    )
    activation_energy_window_seconds = _finite(
        data.get("activation_energy_window_seconds", 0.1),
        "training.activation_energy_window_seconds",
        positive=True,
    )
    if activation_energy_window_seconds > training_window_seconds:
        raise ValidationError(
            "training.activation_energy_window_seconds must be no larger than "
            "training.training_window_seconds"
        )
    try:
        resolve_stft_dimensions(
            device.sample_rate_hz,
            training_window_seconds,
            stft_window_seconds,
            stft_hop_seconds,
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    return TrainingConfig(
        epochs=_integer(data.get("epochs", 30), "training.epochs", minimum=1),
        batch_size=_integer(data.get("batch_size", 128), "training.batch_size", minimum=1),
        learning_rate=_finite(
            data.get("learning_rate", 1e-4), "training.learning_rate", positive=True
        ),
        validation_fraction=validation_fraction,
        training_window_seconds=training_window_seconds,
        dataset_stride_seconds=_finite(
            data.get("dataset_stride_seconds", 0.005),
            "training.dataset_stride_seconds",
            positive=True,
        ),
        stft_window_seconds=stft_window_seconds,
        stft_hop_seconds=stft_hop_seconds,
        activation_energy_window_seconds=activation_energy_window_seconds,
        activation_reference_quantile=activation_reference_quantile,
        activation_smoothing_threshold=activation_smoothing_threshold,
        activation_loss_weight=_finite(
            data.get("activation_loss_weight", 1.0),
            "training.activation_loss_weight",
        ),
        weight_decay=_finite(data.get("weight_decay", 1e-4), "training.weight_decay"),
        normalization=_enum(
            data.get(
                "normalization",
                (
                    NormalizationMode.SIGNED_8BIT
                    if device.kind in {DeviceKind.MYO_BLE, DeviceKind.MYO_DONGLE}
                    else NormalizationMode.DATASET_STANDARDIZE
                ),
            ),
            NormalizationMode,
            "training.normalization",
        ),
        seed=_integer(data.get("seed", 42), "training.seed"),
        export_onnx=_boolean(data.get("export_onnx", True), "training.export_onnx"),
    )


def _parse_inference(raw: object) -> InferenceConfig:
    """Validate backend policy and real-time confidence/debounce settings."""
    data = _object(raw, "inference")
    _only(
        data,
        {
            "backend",
            "confidence_gate",
            "inference_period_seconds",
            "switch_predictions",
            "idle_poll_seconds",
            "maximum_wait_seconds",
        },
        "inference",
    )
    backend = _enum(
        data.get("backend", InferenceBackend.AUTO), InferenceBackend, "inference.backend"
    )
    gate = _finite(data.get("confidence_gate", 0.6), "inference.confidence_gate")
    if not 0 <= gate <= 1:
        raise ValidationError("inference.confidence_gate must be between 0 and 1")
    return InferenceConfig(
        backend,
        gate,
        _finite(
            data.get("inference_period_seconds", 1 / 60),
            "inference.inference_period_seconds",
            positive=True,
        ),
        _integer(data.get("switch_predictions", 3), "inference.switch_predictions", minimum=1),
        _finite(data.get("idle_poll_seconds", 0.002), "inference.idle_poll_seconds", positive=True),
        _finite(
            data.get("maximum_wait_seconds", 0.01), "inference.maximum_wait_seconds", positive=True
        ),
    )


def _parse_dashboard(raw: object) -> DashboardConfig:
    """Validate dashboard bind settings."""
    data = _object(raw, "dashboard")
    _only(data, {"host", "port"}, "dashboard")
    port = _integer(data.get("port", 8765), "dashboard.port", minimum=1)
    if port > 65535:
        raise ValidationError("dashboard.port must be <= 65535")
    return DashboardConfig(
        _string(data.get("host", "127.0.0.1"), "dashboard.host"),
        port,
    )


def _parse_handi(raw: object | None) -> HandiConfig | None:
    """Validate optional Handi control, including verified limits and pose mappings."""
    if raw is None:
        return None
    data = _object(raw, "handi")
    _only(
        data,
        {
            "enabled",
            "rpc_socket",
            "rpc_timeout_seconds",
            "step",
            "openness_step",
            "joints",
            "grips",
            "gesture_mapping",
        },
        "handi",
    )
    joints: list[JointLimit] = []
    for item in cast(list[object], data.get("joints", [])):
        joint = _object(item, "joint")
        _only(joint, {"name", "minimum", "maximum", "start", "open_position"}, "joint")
        minimum = _finite(joint.get("minimum"), "joint.minimum")
        maximum = _finite(joint.get("maximum"), "joint.maximum")
        start = _finite(joint.get("start"), "joint.start")
        open_position = _finite(joint.get("open_position"), "joint.open_position")
        if minimum >= maximum or not minimum <= start <= maximum:
            raise ValidationError("joint requires minimum < maximum and an in-range start")
        if not minimum <= open_position <= maximum:
            raise ValidationError("joint.open_position must be within minimum/maximum")
        joints.append(
            JointLimit(
                _string(joint.get("name"), "joint.name"), minimum, maximum, start, open_position
            )
        )
    grips: list[GripPreset] = []
    for item in cast(list[object], data.get("grips", [])):
        grip = _object(item, "grip")
        _only(grip, {"name", "positions", "led_frame"}, "grip")
        positions = _object(grip.get("positions", {}), "grip.positions")
        grips.append(
            GripPreset(
                _string(grip.get("name"), "grip.name"),
                tuple((name, _finite(value, f"grip.{name}")) for name, value in positions.items()),
                _led_frame(grip.get("led_frame"), "grip.led_frame"),
            )
        )
    mapping = _object(
        data.get("gesture_mapping", {"open": "open", "close": "close"}), "gesture_mapping"
    )
    config = HandiConfig(
        enabled=bool(data.get("enabled", False)),
        rpc_socket=_string(
            data.get("rpc_socket", "/var/run/arduino-router.sock"), "handi.rpc_socket"
        ),
        rpc_timeout_seconds=_finite(
            data.get("rpc_timeout_seconds", 1), "handi.rpc_timeout_seconds", positive=True
        ),
        step=_finite(data.get("step", 5), "handi.step", positive=True),
        openness_step=_finite(
            data.get("openness_step", 0.1), "handi.openness_step", positive=True
        ),
        joints=tuple(joints),
        grips=tuple(grips),
        gesture_mapping=tuple(
            (_string(key, "gesture"), _string(value, "action")) for key, value in mapping.items()
        ),
    )
    if config.enabled and not config.joints:
        raise ValidationError("enabled Handi control requires at least one joint")
    return config


def load_profile(path: str | Path) -> QGripProfile:
    """Load one schema-version-1 JSON profile into a resolved immutable value.

    ``path`` may be relative or use ``~``. All supported defaults are filled,
    paths are made absolute relative to that file, and every section is checked
    before the result is returned. Raises :class:`ValidationError` for unreadable
    JSON, unknown keys, unsupported versions, or invalid values.
    """
    resolved = Path(path).expanduser().resolve()
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read profile {resolved}: {exc}") from exc
    data = _object(raw, "profile")
    _only(
        data,
        {
            "schema_version",
            "data_root",
            "assets_root",
            "device",
            "acquisition",
            "sgt",
            "model",
            "training",
            "inference",
            "dashboard",
            "handi",
        },
        "profile",
    )
    version = _integer(data.get("schema_version", 0), "schema_version", minimum=1)
    if version != SCHEMA_VERSION:
        raise ValidationError(f"unsupported schema_version {version}; expected {SCHEMA_VERSION}")
    base = resolved.parent
    device = _parse_device(data.get("device", {}))
    return QGripProfile(
        version,
        resolved,
        _resolve(base, data.get("data_root", "data"), "data_root"),
        _resolve(base, data.get("assets_root", "assets/images"), "assets_root"),
        device,
        _parse_acquisition(data.get("acquisition", {})),
        _parse_sgt(data.get("sgt", {})),
        _parse_model(data.get("model", {})),
        _parse_training(data.get("training", {}), device),
        _parse_inference(data.get("inference", {})),
        _parse_dashboard(data.get("dashboard", {})),
        _parse_handi(data.get("handi")),
    )


def profile_document(profile: QGripProfile) -> dict[str, object]:
    """Serialize a resolved profile to its schema-v1 JSON-compatible form.

    Data and asset roots are converted back to paths relative to ``profile.path``
    when possible; immutable tuple representations become JSON arrays/objects.
    The returned mapping contains no ``path`` key because it is loader context,
    not part of the profile schema.
    """
    document = asdict(profile)
    document.pop("path")
    document["data_root"] = os.path.relpath(profile.data_root, profile.path.parent)
    document["assets_root"] = os.path.relpath(profile.assets_root, profile.path.parent)
    handi = cast(dict[str, Any] | None, document.get("handi"))
    model = cast(dict[str, Any], document["model"])
    model["architecture"] = dict(model["architecture"])
    if handi is not None:
        handi["gesture_mapping"] = dict(handi["gesture_mapping"])
        for grip in cast(list[dict[str, Any]], handi["grips"]):
            grip["positions"] = dict(grip["positions"])
    return cast(dict[str, object], document)


def write_profile_atomic(profile: QGripProfile, output: str | Path) -> Path:
    """Validate and atomically replace ``output`` with a serialized profile.

    The function writes and fsyncs a sibling temporary file before replacement,
    then reloads the result.  Existing data is therefore either the previous
    profile or a fully valid new profile, never a partial JSON document.
    """
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(profile_document(profile), indent=2) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    load_profile(target)
    return target


def default_profile(kind: DeviceKind | str = DeviceKind.SYNTHETIC) -> dict[str, object]:
    """Return the complete editable schema-v1 document for a device kind.

    Defaults target the synthetic source, use 200 Hz except for SiFi's 1000 Hz
    EMG rate (with an explicit 100 Hz IMU rate), and select signed 8-bit
    normalization for Myo transports.  The mapping is suitable for JSON
    serialization but has not been path-resolved or loaded.
    """
    try:
        device_kind = DeviceKind(kind)
    except ValueError as exc:
        raise ValidationError(f"unsupported device kind: {kind}") from exc
    device: dict[str, object] = {
        "kind": device_kind,
        "sample_rate_hz": 1000 if device_kind == DeviceKind.SIFI else 200,
        "channels": 8,
    }
    if device_kind == DeviceKind.SIFI:
        device["imu_sample_rate_hz"] = ImuConfiguration().sample_rate_hz
    return {
        "schema_version": 1,
        "data_root": "data",
        "assets_root": "assets/images",
        "device": device,
        "acquisition": {
            "ring_buffer_seconds": 10.0,
            "ack_timeout_seconds": 2.0,
            "capture_log_enabled": True,
            "capture_frame_target_bytes": 1 << 20,
            "capture_flush_interval_seconds": 1.0,
            "capture_compression_level": None,
            "capture_fsync_on_boundary": True,
            "health": {
                "window_seconds": 5.0,
                "stale_after_seconds": 2.0,
                "minimum_rate_ratio": 0.9,
                "maximum_rate_ratio": 1.1,
                "maximum_missing_fraction": 0.0,
                "maximum_lost_samples": 0,
            },
        },
        "sgt": {
            "gestures": [
                "rest",
                "close",
                "open",
                "wrist_flexion",
                "wrist_extension",
                "pronation",
                "supination",
            ],
            "trials": 3,
            "duration_seconds": 4,
            "preparation_seconds": 3,
            "practice": True,
            "proportional": True,
            "progress_interval_seconds": 0.05,
            "activation_levels": list(DEFAULT_ACTIVATION_LEVELS),
            "activation_tolerance": 0.1,
            "activation_smoothing_seconds": 0.1,
            "calibration_rest_seconds": 4,
            "calibration_max_seconds": 4,
        },
        "model": {"name": "transformer", "architecture": {}},
        "training": {
            "epochs": 30,
            "batch_size": 128,
            "learning_rate": 1e-4,
            "validation_fraction": 0.2,
            "training_window_seconds": 1.0,
            "dataset_stride_seconds": 0.005,
            "stft_window_seconds": 0.1,
            "stft_hop_seconds": 0.02,
            "activation_energy_window_seconds": 0.1,
            "activation_reference_quantile": 0.9,
            "activation_smoothing_threshold": 0.25,
            "activation_loss_weight": 1.0,
            "weight_decay": 1e-4,
            "normalization": (
                "signed_8bit"
                if device_kind in {DeviceKind.MYO_BLE, DeviceKind.MYO_DONGLE}
                else "dataset_standardize"
            ),
            "seed": 42,
            "export_onnx": True,
        },
        "inference": {
            "backend": "auto",
            "confidence_gate": 0.6,
            "inference_period_seconds": 1 / 60,
            "switch_predictions": 3,
            "idle_poll_seconds": 0.002,
            "maximum_wait_seconds": 0.01,
        },
        "dashboard": {"host": "127.0.0.1", "port": 8765},
        "handi": {
            "enabled": True,
            "rpc_socket": "/var/run/arduino-router.sock",
            "rpc_timeout_seconds": 5,
            "step": 60,
            "openness_step": 0.05,
            "joints": [
                {
                    "name": "thumb_rotate",
                    "minimum": 2244,
                    "maximum": 2840,
                    "start": 2244,
                    "open_position": 2244,
                },
                {
                    "name": "thumb_flex",
                    "minimum": 2100,
                    "maximum": 2944,
                    "start": 2100,
                    "open_position": 2100,
                },
                {
                    "name": "index",
                    "minimum": 2180,
                    "maximum": 3764,
                    "start": 2180,
                    "open_position": 2180,
                },
                {
                    "name": "middle",
                    "minimum": 2173,
                    "maximum": 3841,
                    "start": 2173,
                    "open_position": 2173,
                },
                {
                    "name": "ring",
                    "minimum": 2103,
                    "maximum": 3783,
                    "start": 2103,
                    "open_position": 2103,
                },
                {
                    "name": "baby",
                    "minimum": 2091,
                    "maximum": 3706,
                    "start": 2091,
                    "open_position": 2091,
                },
            ],
            "grips": [
                {
                    "name": "precision_pinch",
                    "positions": {
                        "thumb_rotate": 2804,
                        "thumb_flex": 2695,
                        "index": 2994,
                        "middle": 2449,
                        "ring": 2355,
                        "baby": 2333,
                    },
                    "led_frame": [
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0,
                        0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0,
                        0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0,
                        0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0,
                        0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0,
                        0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    ],
                },
                {
                    "name": "3_jaw_chuck",
                    "positions": {
                        "thumb_rotate": 2840,
                        "thumb_flex": 2704,
                        "index": 2961,
                        "middle": 3249,
                        "ring": 2223,
                        "baby": 2173,
                    },
                    "led_frame": [
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0,
                        0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    ],
                },
                {
                    "name": "column_grip",
                    "positions": {
                        "thumb_rotate": 2840,
                        "thumb_flex": 2592,
                        "index": 3606,
                        "middle": 3674,
                        "ring": 3615,
                        "baby": 3544,
                    },
                    "led_frame": [
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 7, 7, 7, 7, 7, 0, 0, 0, 7, 0, 0,
                        0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0,
                        0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0,
                        0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0,
                        0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0,
                        0, 0, 7, 0, 0, 0, 7, 7, 7, 7, 7, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    ],
                },
                {
                    "name": "lateral_key_grip",
                    "positions": {
                        "thumb_rotate": 2392,
                        "thumb_flex": 2848,
                        "index": 3447,
                        "middle": 3745,
                        "ring": 3615,
                        "baby": 3544,
                    },
                    "led_frame": [
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0,
                        0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0,
                        0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0,
                        0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0,
                        0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0,
                        0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    ],
                },
            ],
            "gesture_mapping": {
                "open": "open",
                "close": "close",
                "wrist_flexion": "next",
                "wrist_extension": "prev",
            },
        },
    }
