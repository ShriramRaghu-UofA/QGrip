import tempfile
import threading
import time
import unittest
from pathlib import Path

import pyarrow.parquet as pq
from sifi_streamer.capture import CaptureLogWriter

from qgrip.artifacts import export_capture, read_capture
from qgrip.domain import ModelName, SGTRequest, TrainingRequest
from qgrip.profiles import load_profile
from qgrip.workflows import InferenceService, SGTService, TrainingService, WorkflowCoordinator
from tests.helpers import write_profile


class SyntheticWorkflowTests(unittest.TestCase):
    def test_export_uses_accepted_presentation_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "session.capture.jsonl.zst"
            attributes = {
                "subject": "subject",
                "device": "synthetic",
                "sample_rate_hz": 200.0,
                "channels": 8,
                "classes": "rest,close",
                "proportional": True,
            }
            packet = {
                "packet_type": "emg_armband",
                "timestamps": [1.0],
                "data": {f"emg{index}": [float(index)] for index in range(8)},
            }
            with CaptureLogWriter(capture, "session", attributes) as writer:
                writer.start_segment("trial-001", "trial", {"trial": 1})
                writer.start_segment(
                    "presentation-001",
                    "presentation",
                    {"label": "rest", "trial": 1, "activation": 0.0},
                )
                writer.append_packet(packet)
                writer.stop_segment("presentation-001", "completed")
                writer.append_marker("presentation-001", "presentation_superseded")
                writer.start_segment(
                    "presentation-002",
                    "presentation",
                    {"label": "close", "trial": 1, "activation": 0.4},
                )
                writer.append_packet(packet)
                writer.stop_segment("presentation-002", "completed")
                writer.stop_segment("trial-001", "completed")
            table = pq.read_table(export_capture(capture))
            self.assertEqual(table.num_rows, 1)
            self.assertEqual(table.column("gesture")[0].as_py(), "close")

    def test_capture_export_train_and_infer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = load_profile(write_profile(Path(directory)))
            capture = SGTService().run(SGTRequest("subject-1", profile, True), threading.Event())
            metadata, rows = read_capture(capture)
            rows = list(rows)
            self.assertTrue(metadata.complete)
            self.assertGreater(len(rows), 0)
            self.assertTrue(capture.name.endswith(".capture.jsonl.zst"))
            self.assertIn("capture_started", {row["record_type"] for row in rows})
            self.assertIn("segment_started", {row["record_type"] for row in rows})
            self.assertTrue(export_capture(capture).is_file())
            checkpoint = TrainingService().train(
                TrainingRequest("subject-1", profile, (), ModelName.DENSE, True), threading.Event()
            )
            prediction = InferenceService(checkpoint).predict(
                tuple(tuple(0.0 for _ in range(8)) for _ in range(4))
            )
            self.assertIn(prediction.gesture, profile.sgt.gestures)
            self.assertGreaterEqual(prediction.activation, 0)
            self.assertEqual(InferenceService(checkpoint).backend, "onnx")

            coordinator = WorkflowCoordinator()
            try:
                coordinator.start_inference(checkpoint, profile)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    status = coordinator.status
                    if status and status.prediction:
                        break
                    time.sleep(0.01)
                self.assertIsNotNone(coordinator.status)
                assert coordinator.status is not None
                self.assertIsNotNone(coordinator.status.prediction)
            finally:
                coordinator.close()

    def test_discrete_model_returns_full_activation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = load_profile(write_profile(Path(directory)))
            capture = SGTService().run(SGTRequest("s", profile, False), threading.Event())
            checkpoint = TrainingService().train(
                TrainingRequest("s", profile, (export_capture(capture),), ModelName.DENSE, False),
                threading.Event(),
            )
            prediction = InferenceService(checkpoint).predict(
                tuple(tuple(0.0 for _ in range(8)) for _ in range(2))
            )
            self.assertEqual(prediction.activation, 1.0)
