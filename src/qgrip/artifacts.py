"""Authoritative capture logs and derived artifact discovery/export."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from qgrip.domain import ArtifactMetadata, QGripProfile
from qgrip.errors import ArtifactError, ValidationError


def validate_subject(subject: str) -> str:
    cleaned = subject.strip()
    if not cleaned or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in cleaned
    ):
        raise ValidationError("subject may contain only letters, numbers, '-' and '_'")
    return cleaned


def subject_root(profile: QGripProfile, subject: str) -> Path:
    return profile.data_root / validate_subject(subject)


def new_capture_path(profile: QGripProfile, subject: str) -> Path:
    raw = subject_root(profile, subject) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    path = raw / f"capture-{stamp}.jsonl"
    if path.exists():
        raise ArtifactError(f"capture already exists: {path}")
    return path


def write_capture_header(stream: Any, metadata: ArtifactMetadata) -> None:
    document = asdict(metadata)
    document["path"] = str(metadata.path)
    document["packet_type"] = "metadata"
    stream.write(json.dumps(document) + "\n")


def read_capture(path: str | Path) -> tuple[ArtifactMetadata, list[dict[str, object]]]:
    source = Path(path).resolve()
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
        header = cast(dict[str, object], json.loads(lines[0]))
        rows = [cast(dict[str, object], json.loads(line)) for line in lines[1:]]
        metadata = ArtifactMetadata(
            source,
            str(header["kind"]),
            str(header["subject"]),
            str(header["created_at"]),
            cast(Any, header["device"]),
            float(cast(float, header["sample_rate_hz"])),
            int(cast(int, header["channels"])),
            tuple(cast(list[str], header["classes"])),
            bool(header["proportional"]),
            bool(header.get("complete", True)),
        )
    except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError, TypeError) as exc:
        raise ArtifactError(f"invalid capture {source}: {exc}") from exc
    return metadata, rows


def export_capture(path: str | Path) -> Path:
    metadata, packets = read_capture(path)
    output = metadata.path.with_suffix(".parquet")
    if output.exists():
        raise ArtifactError(f"derived export already exists: {output}")
    records: list[dict[str, object]] = []
    for packet in packets:
        gesture = str(packet.get("gesture", "unknown"))
        activation = float(cast(float, packet.get("activation", 1.0)))
        timestamp = float(cast(float, packet.get("timestamp", 0)))
        samples = cast(list[list[float]], packet.get("samples", []))
        for sample_index, sample in enumerate(samples):
            record: dict[str, object] = {
                "timestamp": timestamp + sample_index / metadata.sample_rate_hz,
                "gesture": gesture,
                "activation": activation,
            }
            record.update({f"channel_{index}": value for index, value in enumerate(sample)})
            records.append(record)
    if not records:
        raise ArtifactError(f"capture has no signal rows: {metadata.path}")
    table = pa.Table.from_pylist(records)
    temporary = output.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary)
    temporary.replace(output)
    return output


def discover_artifacts(profile: QGripProfile, subject: str | None = None) -> tuple[Path, ...]:
    roots = [profile.data_root]
    legacy = profile.path.parent / "Data"
    if legacy.exists():
        roots.append(legacy)
    found: list[Path] = []
    for root in roots:
        scope = root / subject if subject else root
        if scope.exists():
            found.extend(
                path.resolve()
                for pattern in ("*.jsonl", "*.parquet", "*.pt", "*.onnx")
                for path in scope.rglob(pattern)
            )
    return tuple(sorted(set(found), key=lambda item: item.stat().st_mtime, reverse=True))


def latest_capture(profile: QGripProfile, subject: str) -> Path:
    captures = [
        path
        for path in discover_artifacts(profile, subject)
        if path.suffix in {".jsonl", ".parquet"}
    ]
    if not captures:
        raise ArtifactError(f"no completed capture found for {subject}")
    return captures[0]
