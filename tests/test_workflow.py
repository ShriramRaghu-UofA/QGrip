import tempfile
import threading
import time
import unittest
from pathlib import Path

from qgrip.artifacts import export_capture, read_capture
from qgrip.domain import SGTRequest, TrainingRequest
from qgrip.profiles import load_profile
from qgrip.workflows import InferenceService, SGTService, TrainingService, WorkflowCoordinator
from tests.helpers import write_profile


class SyntheticWorkflowTests(unittest.TestCase):
    def test_capture_export_train_and_infer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = load_profile(write_profile(Path(directory)))
            capture = SGTService().run(SGTRequest("subject-1", profile, True), threading.Event())
            metadata, rows = read_capture(capture)
            self.assertTrue(metadata.complete)
            self.assertGreater(len(rows), 0)
            parquet = export_capture(capture)
            checkpoint = TrainingService(
                epochs=1, batch_size=16, window_seconds=0.05, export_onnx=True
            ).train(
                TrainingRequest("subject-1", profile, (parquet,), "dense", True), threading.Event()
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
            checkpoint = TrainingService(
                epochs=1, batch_size=16, window_seconds=0.05, export_onnx=False
            ).train(
                TrainingRequest("s", profile, (export_capture(capture),), "dense", False),
                threading.Event(),
            )
            prediction = InferenceService(checkpoint).predict(
                tuple(tuple(0.0 for _ in range(8)) for _ in range(2))
            )
            self.assertEqual(prediction.activation, 1.0)
