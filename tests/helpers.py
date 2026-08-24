import json
from pathlib import Path


def write_profile(root: Path, *, handi: bool = False) -> Path:
    document: dict[str, object] = {
        "schema_version": 1,
        "data_root": "data",
        "assets_root": "assets",
        "device": {"kind": "synthetic", "sample_rate_hz": 200, "channels": 8, "seed": 1},
        "acquisition": {
            "ring_buffer_seconds": 10,
            "ack_timeout_seconds": 2,
            "capture_log_enabled": True,
            "capture_frame_target_bytes": 1048576,
            "capture_flush_interval_seconds": 1,
            "capture_compression_level": None,
            "capture_fsync_on_boundary": True,
            "health": {
                "window_seconds": 5,
                "stale_after_seconds": 2,
                "minimum_rate_ratio": 0.9,
                "maximum_rate_ratio": 1.1,
                "maximum_missing_fraction": 0,
                "maximum_lost_samples": 0,
            },
        },
        "sgt": {
            "gestures": ["rest", "open", "close"],
            "trials": 2,
            "duration_seconds": 0.05,
            "preparation_seconds": 0.02,
            "practice": False,
            "proportional": True,
            "progress_interval_seconds": 0.01,
            "activation_levels": [0.25, 0.5, 0.75, 1.0],
            "activation_tolerance": 0.1,
            "activation_smoothing_seconds": 0.05,
            "calibration_rest_seconds": 0.01,
            "calibration_max_seconds": 0.01,
        },
        "model": {"name": "dense", "architecture": {}},
        "training": {
            "epochs": 1,
            "batch_size": 16,
            "learning_rate": 0.0001,
            "validation_fraction": 0.2,
            "training_window_seconds": 0.05,
            "dataset_stride_seconds": 0.005,
            "stft_window_seconds": 0.04,
            "stft_hop_seconds": 0.01,
            "activation_energy_window_seconds": 0.05,
            "activation_reference_quantile": 0.9,
            "activation_smoothing_threshold": 0.25,
            "activation_loss_weight": 1.0,
            "weight_decay": 0.0001,
            "normalization": "dataset_standardize",
            "seed": 42,
            "export_onnx": True,
        },
        "inference": {
            "backend": "auto",
            "confidence_gate": 0,
            "inference_period_seconds": 0.01,
            "switch_predictions": 1,
            "idle_poll_seconds": 0.002,
            "maximum_wait_seconds": 0.01,
        },
        "dashboard": {"host": "127.0.0.1", "port": 8765},
    }
    if handi:
        document["handi"] = {
            "enabled": True,
            "rpc_socket": "/tmp/router.sock",
            "rpc_timeout_seconds": 0.1,
            "step": 5,
            "joints": [
                {"name": "thumb", "minimum": 10, "maximum": 20, "start": 15, "open_position": 10},
                {
                    "name": "index",
                    "minimum": 100,
                    "maximum": 200,
                    "start": 150,
                    "open_position": 100,
                },
            ],
            "grips": [
                {
                    "name": "pinch",
                    "positions": {"thumb": 19, "index": 180},
                    "led_frame": [0] * 104,
                }
            ],
            "gesture_mapping": {"open": "open", "close": "close", "flexion": "pinch"},
        }
    path = root / "profile.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path
