import json
import math
import tempfile
import unittest
from pathlib import Path

from qgrip.domain import NormalizationMode
from qgrip.errors import ValidationError
from qgrip.profiles import load_profile, write_profile_atomic
from tests.helpers import write_profile


class ProfileTests(unittest.TestCase):
    def test_relative_paths_and_atomic_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = load_profile(write_profile(root, handi=True))
            self.assertEqual(profile.data_root, root / "data")
            output = root / "calibrated.json"
            write_profile_atomic(profile, output)
            self.assertEqual(load_profile(output).handi, profile.handi)

    def test_inference_period_and_prediction_debounce_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = load_profile(write_profile(Path(directory)))
            self.assertEqual(profile.inference.inference_period_seconds, 0.01)
            self.assertEqual(profile.inference.switch_predictions, 1)

    def test_training_configuration_is_loaded_from_the_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = load_profile(write_profile(Path(directory)))
            self.assertEqual(profile.training.dataset_stride_seconds, 0.005)
            self.assertEqual(profile.training.training_window_seconds, 0.05)
            self.assertEqual(profile.training.activation_energy_window_seconds, 0.05)
            self.assertEqual(profile.training.activation_reference_quantile, 0.9)
            self.assertEqual(profile.training.activation_smoothing_threshold, 0.25)
            self.assertEqual(profile.training.epochs, 1)

    def test_nested_acquisition_and_model_configuration_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = load_profile(write_profile(Path(directory)))
            self.assertEqual(profile.acquisition.ring_buffer_seconds, 10)
            self.assertEqual(profile.acquisition.health.maximum_lost_samples, 0)
            self.assertEqual(profile.model.architecture, ())

    def test_normalization_defaults_by_device(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_profile(Path(directory))
            document = json.loads(path.read_text(encoding="utf-8"))
            training = document["training"]
            assert isinstance(training, dict)
            training.pop("normalization")
            document["device"] = {"kind": "myo_ble", "sample_rate_hz": 200, "channels": 8}
            path.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(
                load_profile(path).training.normalization, NormalizationMode.SIGNED_8BIT
            )
            document["device"] = {"kind": "sifi", "sample_rate_hz": 1600, "channels": 8}
            path.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(
                load_profile(path).training.normalization,
                NormalizationMode.DATASET_STANDARDIZE,
            )

    def test_unknown_and_nonfinite_values_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_profile(Path(directory))
            document = json.loads(path.read_text(encoding="utf-8"))
            document["surprise"] = True
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_profile(path)
            document.pop("surprise")
            document["device"]["sample_rate_hz"] = math.inf
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_profile(path)

    def test_activation_training_ranges_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_profile(Path(directory))
            document = json.loads(path.read_text(encoding="utf-8"))
            document["training"]["activation_reference_quantile"] = 1
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "reference_quantile"):
                load_profile(path)

            document["training"]["activation_reference_quantile"] = 0.9
            document["training"]["activation_smoothing_threshold"] = 1.1
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "smoothing_threshold"):
                load_profile(path)

            document["training"]["activation_smoothing_threshold"] = 0.25
            document["training"]["activation_energy_window_seconds"] = 0.06
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "energy_window_seconds"):
                load_profile(path)

    def test_incompatible_myo_options_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_profile(Path(directory))
            document = json.loads(path.read_text(encoding="utf-8"))
            document["device"] = {"kind": "myo_ble", "port": "COM4"}
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "cannot define"):
                load_profile(path)

    def test_rest_gesture_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_profile(Path(directory))
            document = json.loads(path.read_text(encoding="utf-8"))
            document["sgt"]["gestures"] = ["open", "close"]
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "include rest"):
                load_profile(path)
