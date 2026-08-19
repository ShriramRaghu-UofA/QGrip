import json
import math
import tempfile
import unittest
from pathlib import Path

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

    def test_incompatible_myo_options_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_profile(Path(directory))
            document = json.loads(path.read_text(encoding="utf-8"))
            document["device"] = {"kind": "myo_ble", "port": "COM4"}
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "cannot define"):
                load_profile(path)
