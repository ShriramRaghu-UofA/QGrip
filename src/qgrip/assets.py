"""Explicit, checksum-verified gesture asset download."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from qgrip.errors import ArtifactError

LIBEMG_GESTURES_COMMIT = "3d7dd6dcdb9fd4563210bece5739c3222d743e97"
LIBEMG_GESTURES_URL = "https://github.com/LibEMG/LibEMGGestures"


def download_assets(target: Path, manifest: dict[str, str] | None = None) -> int:
    """Download an explicit manifest; QGrip intentionally ships no gesture images."""
    target.mkdir(parents=True, exist_ok=True)
    items = manifest or {}
    for relative, expected in items.items():
        if ".." in Path(relative).parts:
            raise ArtifactError(f"unsafe asset path: {relative}")
        url = f"https://raw.githubusercontent.com/LibEMG/LibEMGGestures/{LIBEMG_GESTURES_COMMIT}/{relative}"
        with urllib.request.urlopen(url, timeout=20) as response:
            content = response.read()
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise ArtifactError(f"checksum mismatch for {relative}")
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)
    provenance = {
        "source": LIBEMG_GESTURES_URL,
        "commit": LIBEMG_GESTURES_COMMIT,
        "files": items,
        "license": (
            "Licensing must be confirmed before redistribution; images were fetched by the user."
        ),
    }
    (target / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return len(items)
