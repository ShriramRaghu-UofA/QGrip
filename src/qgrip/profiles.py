"""Strict versioned profile loading and atomic calibration writes."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from qgrip.domain import (
    DashboardConfig,
    DeviceConfig,
    GripPreset,
    HandiConfig,
    InferenceConfig,
    JointLimit,
    ModelConfig,
    QGripProfile,
    SGTConfig,
)
from qgrip.errors import ValidationError

SCHEMA_VERSION = 1
DEVICE_KINDS = {"sifi", "myo_ble", "myo_dongle", "synthetic"}
MODEL_NAMES = {"transformer", "cnn1d", "cnn2d", "dense"}
BACKENDS = {"auto", "torch", "onnx"}


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValidationError(f"{name} must be an object")
    return cast(dict[str, object], value)


def _only(data: dict[str, object], allowed: set[str], name: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValidationError(f"unknown {name} keys: {', '.join(sorted(unknown))}")


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValidationError(f"{name} must be finite{' and positive' if positive else ''}")
    return result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValidationError(f"{name} must be an integer >= {minimum}")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_string(value: object, name: str) -> str | None:
    return None if value is None else _string(value, name)


def _resolve(base: Path, value: object, name: str) -> Path:
    path = Path(_string(value, name)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _parse_device(raw: object) -> DeviceConfig:
    data = _object(raw, "device")
    _only(data, {"kind", "sample_rate_hz", "channels", "address", "port", "seed"}, "device")
    kind = _string(data.get("kind", "synthetic"), "device.kind")
    if kind not in DEVICE_KINDS:
        raise ValidationError(f"unsupported device.kind: {kind}")
    address = _optional_string(data.get("address"), "device.address")
    port = _optional_string(data.get("port"), "device.port")
    if kind == "myo_ble" and port is not None:
        raise ValidationError("myo_ble cannot define device.port")
    if kind == "myo_dongle" and address is not None:
        raise ValidationError("myo_dongle cannot define device.address")
    return DeviceConfig(
        kind=cast(Any, kind),
        sample_rate_hz=_finite(
            data.get("sample_rate_hz", 200), "device.sample_rate_hz", positive=True
        ),
        channels=_integer(data.get("channels", 8), "device.channels", minimum=1),
        address=address,
        port=port,
        seed=_integer(data.get("seed", 7), "device.seed"),
    )


def _parse_sgt(raw: object) -> SGTConfig:
    data = _object(raw, "sgt")
    _only(
        data,
        {
            "gestures",
            "trials",
            "duration_seconds",
            "practice",
            "proportional",
            "activation_calibration",
        },
        "sgt",
    )
    gestures_raw = data.get("gestures", ["rest", "open", "close"])
    if not isinstance(gestures_raw, list):
        raise ValidationError("sgt.gestures must be a list")
    gestures = tuple(_string(item, "gesture") for item in gestures_raw)
    if len(gestures) < 2 or len(set(gestures)) != len(gestures):
        raise ValidationError("sgt.gestures must contain at least two unique names")
    return SGTConfig(
        gestures=gestures,
        trials=_integer(data.get("trials", 3), "sgt.trials", minimum=1),
        duration_seconds=_finite(
            data.get("duration_seconds", 4), "sgt.duration_seconds", positive=True
        ),
        practice=bool(data.get("practice", True)),
        proportional=bool(data.get("proportional", True)),
        activation_calibration=bool(data.get("activation_calibration", True)),
    )


def _parse_model(raw: object) -> ModelConfig:
    data = _object(raw, "model")
    _only(data, {"name", "preset_version"}, "model")
    name = _string(data.get("name", "transformer"), "model.name")
    if name not in MODEL_NAMES:
        raise ValidationError(f"unsupported model.name: {name}")
    return ModelConfig(
        cast(Any, name), _integer(data.get("preset_version", 1), "model.preset_version", minimum=1)
    )


def _parse_inference(raw: object) -> InferenceConfig:
    data = _object(raw, "inference")
    _only(data, {"backend", "confidence_gate", "interval_seconds", "switch_samples"}, "inference")
    backend = _string(data.get("backend", "auto"), "inference.backend")
    if backend not in BACKENDS:
        raise ValidationError(f"unsupported inference.backend: {backend}")
    gate = _finite(data.get("confidence_gate", 0.6), "inference.confidence_gate")
    if not 0 <= gate <= 1:
        raise ValidationError("inference.confidence_gate must be between 0 and 1")
    return InferenceConfig(
        cast(Any, backend),
        gate,
        _finite(data.get("interval_seconds", 0.05), "inference.interval_seconds", positive=True),
        _integer(data.get("switch_samples", 3), "inference.switch_samples", minimum=1),
    )


def _parse_dashboard(raw: object) -> DashboardConfig:
    data = _object(raw, "dashboard")
    _only(data, {"host", "port", "handi_url"}, "dashboard")
    port = _integer(data.get("port", 8765), "dashboard.port", minimum=1)
    if port > 65535:
        raise ValidationError("dashboard.port must be <= 65535")
    return DashboardConfig(
        _string(data.get("host", "127.0.0.1"), "dashboard.host"),
        port,
        _optional_string(data.get("handi_url"), "dashboard.handi_url"),
    )


def _parse_handi(raw: object | None) -> HandiConfig | None:
    if raw is None:
        return None
    data = _object(raw, "handi")
    _only(
        data,
        {
            "enabled",
            "rpc_socket",
            "rpc_host",
            "rpc_port",
            "rpc_timeout_seconds",
            "api_enabled",
            "api_host",
            "api_port",
            "step",
            "map_flexion_to_grips",
            "joints",
            "grips",
            "gesture_mapping",
        },
        "handi",
    )
    joints: list[JointLimit] = []
    for item in cast(list[object], data.get("joints", [])):
        joint = _object(item, "joint")
        _only(joint, {"name", "minimum", "maximum", "start"}, "joint")
        minimum = _finite(joint.get("minimum"), "joint.minimum")
        maximum = _finite(joint.get("maximum"), "joint.maximum")
        start = _finite(joint.get("start"), "joint.start")
        if minimum >= maximum or not minimum <= start <= maximum:
            raise ValidationError("joint requires minimum < maximum and an in-range start")
        joints.append(JointLimit(_string(joint.get("name"), "joint.name"), minimum, maximum, start))
    grips: list[GripPreset] = []
    for item in cast(list[object], data.get("grips", [])):
        grip = _object(item, "grip")
        _only(grip, {"name", "positions"}, "grip")
        positions = _object(grip.get("positions", {}), "grip.positions")
        grips.append(
            GripPreset(
                _string(grip.get("name"), "grip.name"),
                tuple((name, _finite(value, f"grip.{name}")) for name, value in positions.items()),
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
        rpc_host=_string(data.get("rpc_host", "127.0.0.1"), "handi.rpc_host"),
        rpc_port=_integer(data.get("rpc_port", 5000), "handi.rpc_port", minimum=1),
        rpc_timeout_seconds=_finite(
            data.get("rpc_timeout_seconds", 1), "handi.rpc_timeout_seconds", positive=True
        ),
        api_enabled=bool(data.get("api_enabled", False)),
        api_host=_string(data.get("api_host", "127.0.0.1"), "handi.api_host"),
        api_port=_integer(data.get("api_port", 8770), "handi.api_port", minimum=1),
        step=_finite(data.get("step", 5), "handi.step", positive=True),
        map_flexion_to_grips=bool(data.get("map_flexion_to_grips", False)),
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
            "sgt",
            "model",
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
    return QGripProfile(
        version,
        resolved,
        _resolve(base, data.get("data_root", "data"), "data_root"),
        _resolve(base, data.get("assets_root", "assets/gestures"), "assets_root"),
        _parse_device(data.get("device", {})),
        _parse_sgt(data.get("sgt", {})),
        _parse_model(data.get("model", {})),
        _parse_inference(data.get("inference", {})),
        _parse_dashboard(data.get("dashboard", {})),
        _parse_handi(data.get("handi")),
    )


def profile_document(profile: QGripProfile) -> dict[str, object]:
    document = asdict(profile)
    document.pop("path")
    document["data_root"] = os.path.relpath(profile.data_root, profile.path.parent)
    document["assets_root"] = os.path.relpath(profile.assets_root, profile.path.parent)
    handi = cast(dict[str, Any] | None, document.get("handi"))
    if handi is not None:
        handi["gesture_mapping"] = dict(handi["gesture_mapping"])
        for grip in cast(list[dict[str, Any]], handi["grips"]):
            grip["positions"] = dict(grip["positions"])
    return cast(dict[str, object], document)


def write_profile_atomic(profile: QGripProfile, output: str | Path) -> Path:
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


def default_profile(kind: str = "synthetic") -> dict[str, object]:
    if kind not in DEVICE_KINDS:
        raise ValidationError(f"unsupported device kind: {kind}")
    device: dict[str, object] = {
        "kind": kind,
        "sample_rate_hz": 1600 if kind == "sifi" else 200,
        "channels": 8,
    }
    return {
        "schema_version": 1,
        "data_root": "data",
        "assets_root": "assets/gestures",
        "device": device,
        "sgt": {
            "gestures": ["rest", "open", "close"],
            "trials": 3,
            "duration_seconds": 4,
            "practice": True,
            "proportional": True,
            "activation_calibration": True,
        },
        "model": {"name": "transformer", "preset_version": 1},
        "inference": {
            "backend": "auto",
            "confidence_gate": 0.6,
            "interval_seconds": 0.05,
            "switch_samples": 3,
        },
        "dashboard": {"host": "127.0.0.1", "port": 8765},
    }
