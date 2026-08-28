import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
from sifi_streamer.capture import CaptureLogWriter

from qgrip.capture.artifacts import (
    calibration_path,
    export_capture,
    read_capture,
    write_calibration_atomic,
)
from qgrip.core.domain import (
    JobState,
    ModelName,
    SGTCommand,
    SGTProgress,
    SGTRequest,
    TrainingRequest,
)
from qgrip.core.errors import ArtifactError, ValidationError
from qgrip.core.profiles import load_profile
from qgrip.runtime.workflows import (
    CalibrationService,
    InferenceService,
    ModelSummaryService,
    ProgressCallback,
    SGTCommandGate,
    SGTService,
    TrainingService,
    WorkflowCoordinator,
    _valid_emg_rows,
    run_inference_benchmark,
    run_inference_benchmark_suite,
)
from tests.helpers import write_profile


class SyntheticWorkflowTests(unittest.TestCase):
    def calibrate(self, profile, subject: str) -> None:
        """Create the canonical calibration artifact for proportional SGT tests."""
        write_calibration_atomic(
            calibration_path(profile, subject),
            {
                "version": 1,
                "capture": "synthetic-test-calibration",
                "sample_rate_hz": profile.device.sample_rate_hz,
                "channels": profile.device.channels,
                "rest_floor": 1.0,
                "class_references": {
                    gesture: 10.0 for gesture in profile.sgt.gestures if gesture != "rest"
                },
            },
        )

    def test_live_activation_keeps_complete_channel_rows(self) -> None:
        samples = np.asarray([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        validity = np.asarray([[True, True], [True, False], [True, True]])

        np.testing.assert_array_equal(
            _valid_emg_rows(samples, validity),
            np.asarray([[1.0, 10.0], [3.0, 30.0]]),
        )

    def test_export_uses_accepted_presentation_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "session.capture.jsonl.zst"
            attributes = {
                "subject": "subject",
                "created_at": "2026-01-01T00:00:00+00:00",
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
            table = pq.read_table(export_capture(capture, activation_energy_window_seconds=0.1))
            self.assertEqual(table.num_rows, 1)
            self.assertEqual(table.column("gesture")[0].as_py(), "close")

    def test_export_preserves_held_activation_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "session.capture.jsonl.zst"
            attributes = {
                "subject": "subject",
                "created_at": "2026-01-01T00:00:00+00:00",
                "device": "synthetic",
                "sample_rate_hz": 200.0,
                "channels": 8,
                "classes": "rest,close",
                "proportional": True,
            }

            def packet(stamp: float) -> dict[str, object]:
                return {
                    "packet_type": "emg_armband",
                    "timestamps": [stamp],
                    "data": {f"emg{index}": [float(index)] for index in range(8)},
                }

            with CaptureLogWriter(capture, "session", attributes) as writer:
                writer.start_segment("trial-001", "trial", {"trial": 1})
                writer.start_segment(
                    "presentation-001",
                    "presentation",
                    {"label": "close", "trial": 1, "activation": 1.0},
                )
                for index in range(5):
                    writer.append_packet(packet(float(index)))
                writer.stop_segment("presentation-001", "completed")
                writer.stop_segment("trial-001", "completed")
            table = pq.read_table(export_capture(capture, activation_energy_window_seconds=0.1))
            activations = table.column("activation").to_pylist()
            # A presentation is one held target, not a reconstructed trajectory.
            self.assertEqual([round(value, 3) for value in activations], [1.0] * 5)
            self.assertEqual(table.column_names[-1], "activation_energy")
            self.assertIn("sample_rate_hz", table.column_names)
            self.assertNotIn("sample_rate", table.column_names)
            self.assertEqual(table.column("activation_energy").to_pylist(), [0.0] * 5)
            self.assertEqual(
                table.schema.metadata[b"qgrip.activation_energy.window_samples"], b"20"
            )

    def test_capture_export_train_and_infer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = load_profile(write_profile(Path(directory)))
            self.calibrate(profile, "subject-1")
            capture = SGTService().run(SGTRequest("subject-1", profile, True), threading.Event())
            metadata, rows = read_capture(capture)
            rows = list(rows)
            self.assertTrue(metadata.complete)
            self.assertGreater(len(rows), 0)
            self.assertTrue(capture.name.endswith(".capture.jsonl.zst"))
            self.assertIn("capture_started", {row["record_type"] for row in rows})
            self.assertIn("segment_started", {row["record_type"] for row in rows})
            self.assertTrue(
                export_capture(
                    capture,
                    activation_energy_window_seconds=(
                        profile.training.activation_energy_window_seconds
                    ),
                ).is_file()
            )
            checkpoint = TrainingService().train(
                TrainingRequest("subject-1", profile, (), ModelName.DENSE, True), threading.Event()
            )
            preset_summary = ModelSummaryService.preview(profile, ModelName.DENSE, True)
            self.assertEqual(preset_summary.source, "preset")
            self.assertEqual(preset_summary.model_name, ModelName.DENSE)
            self.assertEqual(preset_summary.labels, profile.sgt.gestures)
            self.assertEqual(preset_summary.channels, profile.device.channels)
            self.assertGreater(preset_summary.parameter_count, 0)
            self.assertEqual(
                preset_summary.parameter_count, preset_summary.trainable_parameter_count
            )
            self.assertIn("DenseEMGClassifier", preset_summary.module_tree)

            checkpoint_summary = ModelSummaryService.checkpoint(checkpoint)
            self.assertEqual(checkpoint_summary.source, "checkpoint")
            self.assertEqual(checkpoint_summary.checkpoint, checkpoint.resolve())
            self.assertEqual(checkpoint_summary.labels, profile.sgt.gestures)
            self.assertIsNotNone(checkpoint_summary.validation_loss)
            self.assertIsNotNone(checkpoint_summary.validation_accuracy)
            self.assertEqual(
                ModelSummaryService.checkpoint(checkpoint.with_suffix(".onnx")),
                checkpoint_summary,
            )
            model_metadata = json.loads(
                (checkpoint.parent / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(model_metadata["checkpoint_version"], 1)
            self.assertNotIn("n_channels", model_metadata)
            self.assertNotIn("channels", model_metadata)
            self.assertNotIn("sample_rate", model_metadata)
            self.assertEqual(model_metadata["model_config"]["n_channels"], 8)
            inference = InferenceService(checkpoint)
            prediction = inference.predict(
                np.zeros((inference.window_size, inference.channels), dtype=np.float32)
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

            benchmark = run_inference_benchmark(inference, iterations=10, warmup=2)
            self.assertEqual(benchmark.model_name, "dense")
            self.assertEqual(benchmark.iterations, 10)
            self.assertEqual(benchmark.warmup, 2)
            self.assertEqual(benchmark.window_size, inference.window_size)
            self.assertEqual(benchmark.channels, inference.channels)
            self.assertIn(benchmark.device, {"cpu", "gpu"})
            self.assertGreater(benchmark.mean_ms, 0)
            self.assertGreaterEqual(benchmark.p99_ms, benchmark.p95_ms)
            self.assertGreaterEqual(benchmark.p95_ms, benchmark.median_ms)
            self.assertGreater(benchmark.throughput_hz, 0)
            with self.assertRaises(ValidationError):
                run_inference_benchmark(inference, iterations=0)
            with self.assertRaises(ValidationError):
                run_inference_benchmark(inference, warmup=-1)

            cpu_inference = InferenceService(checkpoint, "torch", "cpu")
            self.assertEqual(cpu_inference.device, "cpu")
            suite = run_inference_benchmark_suite(checkpoint, iterations=2, warmup=0)
            self.assertIn(("onnx", "cpu"), {(item.backend, item.device) for item in suite})
            self.assertIn(("torch", "cpu"), {(item.backend, item.device) for item in suite})

    def test_model_summary_rejects_missing_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ArtifactError):
            ModelSummaryService.checkpoint(Path(directory) / "missing.pt")

    def test_discrete_model_returns_full_activation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = load_profile(write_profile(Path(directory)))
            capture = SGTService().run(SGTRequest("s", profile, False), threading.Event())
            checkpoint = TrainingService().train(
                TrainingRequest(
                    "s",
                    profile,
                    (
                        export_capture(
                            capture,
                            activation_energy_window_seconds=(
                                profile.training.activation_energy_window_seconds
                            ),
                        ),
                    ),
                    ModelName.DENSE,
                    False,
                ),
                threading.Event(),
            )
            inference = InferenceService(checkpoint)
            prediction = inference.predict(
                np.zeros((inference.window_size, inference.channels), dtype=np.float32)
            )
            self.assertEqual(prediction.activation, 1.0)

    def test_sgt_reports_practice_and_presentation_stimuli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = load_profile(write_profile(Path(directory)))
            self.calibrate(profile, "s")
            profile = replace(profile, sgt=replace(profile.sgt, practice=True))
            progress = []
            SGTService().run(SGTRequest("s", profile, True), threading.Event(), progress.append)
            # A get-ready preparation countdown precedes the first prompt.
            self.assertEqual(progress[0].stage, "preparation")
            self.assertEqual(progress[0].gesture, "rest")
            self.assertIn("Get ready", progress[0].instruction or "")
            self.assertEqual(progress[0].duration_seconds, profile.sgt.preparation_seconds)
            self.assertTrue(
                any(
                    item.stage == "preparation" and item.activation > 0
                    for item in progress
                    if item.gesture != "rest"
                )
            )
            practice = next(item for item in progress if item.stage == "practice")
            self.assertEqual(practice.gesture, "rest")
            self.assertIn("Practice", practice.instruction or "")
            self.assertEqual(practice.duration_seconds, profile.sgt.duration_seconds)
            # The practice stage surfaces the same activation ramp as presentations.
            self.assertTrue(
                any(
                    item.stage == "practice" and item.activation > 0
                    for item in progress
                    if item.gesture != "rest"
                )
            )
            self.assertTrue(
                any(
                    item.stage == "practice" and item.measured_activation > 0
                    for item in progress
                    if item.gesture != "rest"
                )
            )
            self.assertGreater(max(item.elapsed_seconds for item in progress), 0)
            self.assertTrue(any(item.stage == "presentation" for item in progress))

    def test_preparation_precedes_each_recorded_presentation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = load_profile(write_profile(Path(directory)))
            self.calibrate(profile, "s")
            progress: list[SGTProgress] = []
            SGTService().run(SGTRequest("s", profile, False), threading.Event(), progress.append)
            # Collapse the repeated progress ticks into ordered, distinct stage runs.
            runs: list[str | None] = []
            for item in progress:
                if not runs or runs[-1] != item.stage:
                    runs.append(item.stage)
            self.assertIn("preparation", runs)
            # Every recorded presentation run must be introduced by a preparation run.
            self.assertTrue(any(stage == "presentation" for stage in runs))
            for index, stage in enumerate(runs):
                if stage == "presentation":
                    self.assertEqual(runs[index - 1], "preparation")

    def test_preparation_preserves_accumulated_trial_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = load_profile(write_profile(Path(directory)))
            self.calibrate(profile, "s")
            progress: list[SGTProgress] = []
            SGTService().run(SGTRequest("s", profile, False), threading.Event(), progress.append)
            prep_trials = [item.trial for item in progress if item.stage == "preparation"]
            self.assertTrue(len(prep_trials) > 0)
            # Later preparation periods must reflect already completed presentations.
            self.assertTrue(max(prep_trials) > 0)

    def test_preparation_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = load_profile(write_profile(Path(directory)))
            profile = replace(profile, sgt=replace(profile.sgt, preparation_seconds=0.0))
            self.calibrate(profile, "s")
            progress: list[SGTProgress] = []
            SGTService().run(SGTRequest("s", profile, True), threading.Event(), progress.append)
            self.assertFalse(any(item.stage == "preparation" for item in progress))

    def test_coordinator_exposes_sgt_stimulus(self) -> None:
        class ReportingSGT(SGTService):
            def run(
                self,
                request: SGTRequest,
                cancel: threading.Event,
                progress: ProgressCallback | None = None,
                gate: object | None = None,
            ) -> Path:
                assert progress is not None
                progress(
                    SGTProgress(
                        JobState.RUNNING,
                        gesture="rest",
                        stage="practice",
                        instruction="Practice: rest.",
                        stimulus_image="rest.png",
                    )
                )
                cancel.wait(0.2)
                return Path("capture.capture.jsonl.zst")

        with tempfile.TemporaryDirectory() as directory:
            profile = load_profile(write_profile(Path(directory)))
            self.calibrate(profile, "s")
            coordinator = WorkflowCoordinator(sgt=ReportingSGT())
            try:
                coordinator.start_sgt(SGTRequest("s", profile, True))
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    status = coordinator.status
                    if status and status.instruction:
                        break
                    time.sleep(0.01)
                assert coordinator.status is not None
                self.assertEqual(coordinator.status.gesture, "rest")
                self.assertEqual(coordinator.status.stimulus_image, "rest.png")
            finally:
                coordinator.close()

    def test_manual_sgt_gate_waits_and_supports_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = load_profile(write_profile(Path(directory)))
            self.calibrate(profile, "s")
            gate = SGTCommandGate()
            events: list[SGTProgress] = []
            repeated = {"done": False}

            def on_progress(value: SGTProgress) -> None:
                events.append(value)
                if value.awaiting_command:
                    if not repeated["done"]:
                        repeated["done"] = True
                        gate.send(SGTCommand.REPEAT)
                    else:
                        gate.send(SGTCommand.RESUME)

            capture = SGTService().run(
                SGTRequest("s", profile, True, auto=False),
                threading.Event(),
                on_progress,
                gate,
            )
            # Manual mode must pause for the operator after each stimulus.
            self.assertTrue(any(item.awaiting_command for item in events))
            self.assertTrue(repeated["done"])
            # A repeated take is superseded, so the export still projects clean rows.
            table = pq.read_table(export_capture(capture, activation_energy_window_seconds=0.1))
            self.assertGreater(table.num_rows, 0)

    def test_calibration_completion_emits_final_trial_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = load_profile(write_profile(Path(directory)))
            progress: list[SGTProgress] = []
            cal_file = Path(directory) / "cal.json"
            with patch("qgrip.runtime.workflows.derive_calibration", return_value=cal_file):
                CalibrationService().run("s", profile, threading.Event(), progress.append)
            self.assertTrue(len(progress) > 0)
            final = progress[-1]
            self.assertEqual(final.state, JobState.COMPLETED)
            self.assertEqual(final.stage, "calibration")
            self.assertEqual(final.trial, len(profile.sgt.gestures))
            self.assertEqual(final.total_trials, len(profile.sgt.gestures))

    def test_sgt_completion_emits_final_trial_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = load_profile(write_profile(Path(directory)))
            self.calibrate(profile, "s")
            progress: list[SGTProgress] = []
            SGTService().run(SGTRequest("s", profile, False), threading.Event(), progress.append)
            self.assertTrue(len(progress) > 0)
            final = progress[-1]
            self.assertEqual(final.state, JobState.COMPLETED)
            expected_total = profile.sgt.trials * len(profile.sgt.gestures)
            self.assertEqual(final.trial, expected_total)
            self.assertEqual(final.total_trials, expected_total)
