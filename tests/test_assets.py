import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qgrip.capture.assets import GESTURE_ASSETS, GestureAsset, download_assets
from qgrip.core.errors import ArtifactError


class AssetTests(unittest.TestCase):
    def test_downloads_only_selected_gestures_and_writes_manifest(self) -> None:
        content = b"verified gesture image"
        checksum = hashlib.sha256(content).hexdigest()
        assets = {"rest": GestureAsset("Images/No_Motion.png", checksum)}

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "images"
            with (
                patch.dict(GESTURE_ASSETS, assets, clear=True),
                patch(
                    "qgrip.capture.assets.urllib.request.urlopen", return_value=io.BytesIO(content)
                ) as get,
            ):
                self.assertEqual(download_assets(target, ["rest"]), 1)

            self.assertEqual((target / "rest.png").read_bytes(), content)
            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(list(manifest["files"]), ["rest.png"])
            self.assertEqual(manifest["files"]["rest.png"]["sha256"], checksum)
            self.assertIn("citation", manifest)
            get.assert_called_once()

    def test_reuses_an_existing_verified_image(self) -> None:
        content = b"verified gesture image"
        checksum = hashlib.sha256(content).hexdigest()
        assets = {"rest": GestureAsset("Images/No_Motion.png", checksum)}

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / "rest.png").write_bytes(content)
            with (
                patch.dict(GESTURE_ASSETS, assets, clear=True),
                patch("qgrip.capture.assets.urllib.request.urlopen") as get,
            ):
                self.assertEqual(download_assets(target, ["rest"]), 1)
            get.assert_not_called()

    def test_rejects_an_unknown_gesture_before_creating_the_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "images"
            with self.assertRaisesRegex(ArtifactError, "unsupported gesture"):
                download_assets(target, ["not-a-gesture"])
            self.assertFalse(target.exists())
