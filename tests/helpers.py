import json
from pathlib import Path


def write_profile(root: Path, *, handi: bool = False) -> Path:
    document: dict[str, object] = {
        "schema_version": 1,
        "data_root": "data",
        "assets_root": "assets",
        "device": {"kind": "synthetic", "sample_rate_hz": 200, "channels": 8, "seed": 1},
        "sgt": {
            "gestures": ["rest", "open", "close"],
            "trials": 1,
            "duration_seconds": 0.01,
            "practice": False,
            "proportional": True,
            "activation_calibration": True,
        },
        "model": {"name": "dense", "preset_version": 1},
        "inference": {
            "backend": "auto",
            "confidence_gate": 0,
            "interval_seconds": 0.01,
            "switch_samples": 1,
        },
        "dashboard": {"host": "127.0.0.1", "port": 8765},
    }
    if handi:
        document["handi"] = {
            "enabled": True,
            "rpc_socket": "/tmp/router.sock",
            "rpc_timeout_seconds": 0.1,
            "api_enabled": False,
            "step": 5,
            "joints": [
                {"name": "thumb", "minimum": 10, "maximum": 20, "start": 15},
                {"name": "index", "minimum": 100, "maximum": 200, "start": 150},
            ],
            "grips": [{"name": "pinch", "positions": {"thumb": 19, "index": 180}}],
            "gesture_mapping": {"open": "open", "close": "close", "flexion": "pinch"},
        }
    path = root / "profile.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path
