"""Explicit, checksum-verified gesture image downloads."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from qgrip.errors import ArtifactError

LIBEMG_GESTURES_COMMIT = "c17792f1966f23f7dafda7c47a65012e47a2e7ee"
LIBEMG_GESTURES_URL = "https://github.com/LibEMG/LibEMGGestures"
LIBEMG_CITATION_URL = f"{LIBEMG_GESTURES_URL}/tree/{LIBEMG_GESTURES_COMMIT}"


@dataclass(frozen=True, slots=True)
class GestureAsset:
    """Pinned upstream image path and SHA-256 digest for one gesture prompt."""

    source: str
    sha256: str


GESTURE_ASSETS: dict[str, GestureAsset] = {
    "rest": GestureAsset(
        "Images/No_Motion.png",
        "af15f8e9af33420528325b270fd9eba3daec57fd3e3f4e5ea7bcc90dba14c504",
    ),
    "close": GestureAsset(
        "Images/Hand_Close.png",
        "c808c643cb7b857171e50ff0c7aa5d61e39aefc12c25e28a864f48f64f5e60fd",
    ),
    "open": GestureAsset(
        "Images/Hand_Open.png",
        "a5ceeeccca79c4e6900673f2a4f982a30dc4d75da1eed1622dceee1358c85777",
    ),
    "wrist_flexion": GestureAsset(
        "Images/Wrist_Flexion.png",
        "de8428a0320edb1dcd6c7618670a983a9fae032847b77f19b0f7948ab269e927",
    ),
    "wrist_extension": GestureAsset(
        "Images/Wrist_Extension.png",
        "b3933d029f875d9fc9eb88f8a904d3769667d7d0be03043f9fbed57f9afe36ac",
    ),
    "pronation": GestureAsset(
        "Images/Pronation.png",
        "0259b1804eafc3ef8469004f7410d2e4f771b5f5d2fbc07bc2db9c21b5847425",
    ),
    "supination": GestureAsset(
        "Images/Supination.png",
        "359cfbb1e15b7f058e704a1c102d363d57eae246b8c711c33402bc7bed764495",
    ),
}
DEFAULT_GESTURES = tuple(GESTURE_ASSETS)


def download_assets(target: Path, gestures: Sequence[str] = DEFAULT_GESTURES) -> int:
    """Ensure that selected, pinned LibEMG gesture images exist in ``target``."""
    selected = tuple(dict.fromkeys(gestures))
    unknown = sorted(set(selected) - GESTURE_ASSETS.keys())
    if unknown:
        supported = ", ".join(GESTURE_ASSETS)
        raise ArtifactError(f"unsupported gesture(s): {', '.join(unknown)}; supported: {supported}")

    target.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, str]] = {}
    for gesture in selected:
        asset = GESTURE_ASSETS[gesture]
        output = target / f"{gesture}.png"
        if not output.is_file() or _sha256(output.read_bytes()) != asset.sha256:
            url = (
                "https://raw.githubusercontent.com/LibEMG/LibEMGGestures/"
                f"{LIBEMG_GESTURES_COMMIT}/{asset.source}"
            )
            with urllib.request.urlopen(url, timeout=20) as response:
                content = response.read()
            if _sha256(content) != asset.sha256:
                raise ArtifactError(f"checksum mismatch for {asset.source}")
            output.write_bytes(content)
        files[output.name] = {
            "gesture": gesture,
            "upstream_path": asset.source,
            "sha256": asset.sha256,
        }

    manifest = {
        "source": LIBEMG_GESTURES_URL,
        "commit": LIBEMG_GESTURES_COMMIT,
        "files": files,
        "citation": (
            "LibEMGGestures asks users of its images to cite the work identified in its README."
        ),
        "citation_url": LIBEMG_CITATION_URL,
        "redistribution": "Review upstream attribution and licensing before redistribution.",
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return len(selected)


def _sha256(content: bytes) -> str:
    """Return the lowercase SHA-256 hex digest used for asset integrity checks."""
    return hashlib.sha256(content).hexdigest()
