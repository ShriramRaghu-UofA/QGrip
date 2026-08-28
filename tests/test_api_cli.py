import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from qgrip.core.domain import BenchmarkResult, ComputePreference
from qgrip.runtime.api import create_app, notification_for
from qgrip.runtime.cli import main
from tests.helpers import write_profile


class AdapterTests(unittest.TestCase):
    def test_openapi_and_token_protection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(write_profile(Path(directory)), token="secret")
            with TestClient(app) as client:
                self.assertEqual(client.get("/openapi.json").status_code, 200)
                self.assertEqual(client.get("/api/v1/bootstrap").status_code, 401)
                bootstrap = client.get("/api/v1/bootstrap", headers={"X-QGrip-Token": "secret"})
                self.assertEqual(bootstrap.status_code, 200)
                self.assertEqual(bootstrap.json()["activation_tolerance"], 0.1)
                self.assertTrue(bootstrap.json()["proportional"])
                self.assertEqual(client.get("/api/v1/artifacts").status_code, 401)
                response = client.get("/api/v1/artifacts", headers={"X-QGrip-Token": "secret"})
                self.assertEqual(response.status_code, 200)
                self.assertFalse(response.json()["calibration_ready"])

    def test_stream_requires_a_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(write_profile(Path(directory)), token="secret")
            with TestClient(app) as client:
                self.assertEqual(client.get("/api/v1/stream").status_code, 401)
                self.assertEqual(client.get("/api/v1/stream?token=wrong").status_code, 401)

    def test_stream_frames_use_named_channels(self) -> None:
        # Exercise the event generator directly: consuming it through the live
        # ASGI stream would block on the intentionally endless push loop.
        from qgrip.core.domain import JobState, JobStatus

        completed = JobStatus("job", "training", JobState.COMPLETED)
        failed = JobStatus("job", "training", JobState.FAILED, message="boom")
        self.assertEqual(
            notification_for(completed),
            {"kind": "training", "level": "success", "message": "training completed"},
        )
        failed_notification = notification_for(failed)
        assert failed_notification is not None
        self.assertEqual(failed_notification["level"], "error")
        self.assertEqual(failed_notification["message"], "boom")
        self.assertIsNone(notification_for(JobStatus("job", "training", JobState.RUNNING)))

    def test_cli_help_entry_points(self) -> None:
        with self.assertRaises(SystemExit) as exit_context:
            main(["--help"])
        self.assertEqual(exit_context.exception.code, 0)

    def test_infer_defaults_to_live_with_an_explicit_once_mode(self) -> None:
        from qgrip.runtime.cli import build_parser

        live = build_parser().parse_args(["infer", "model.pt", "--profile", "profile.json"])
        once = build_parser().parse_args(
            ["infer", "model.pt", "--profile", "profile.json", "--once"]
        )
        self.assertFalse(live.once)
        self.assertTrue(once.once)

    def test_benchmark_takes_no_profile_and_has_defaults(self) -> None:
        from qgrip.runtime.cli import build_parser

        parsed = build_parser().parse_args(["benchmark", "model.pt"])
        self.assertEqual(parsed.backend, "auto")
        self.assertEqual(parsed.iterations, 1000)
        self.assertEqual(parsed.warmup, 20)
        self.assertEqual(parsed.device, "gpu")
        self.assertFalse(parsed.json)

    def test_dashboard_benchmark_exposes_backend_device_results(self) -> None:
        result = BenchmarkResult(
            backend="onnx",
            device=ComputePreference.CPU,
            model_name="dense",
            iterations=10,
            warmup=2,
            window_size=200,
            channels=8,
            mean_ms=1.2,
            median_ms=1.1,
            p95_ms=1.5,
            p99_ms=1.6,
            min_ms=1.0,
            max_ms=1.7,
            stdev_ms=0.1,
            throughput_hz=833.3,
        )
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(write_profile(Path(directory)), token="secret")
            with (
                patch(
                    "qgrip.runtime.api.run_inference_benchmark_suite",
                    return_value=(result,),
                ) as benchmark,
                TestClient(app) as client,
            ):
                response = client.post(
                    "/api/v1/benchmark",
                    headers={"X-QGrip-Token": "secret"},
                    json={"model": "model.pt", "iterations": 10, "warmup": 2},
                )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"results": [asdict(result)]})
            benchmark.assert_called_once()
