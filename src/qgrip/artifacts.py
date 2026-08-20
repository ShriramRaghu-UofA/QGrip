"""Authoritative streamer captures and QGrip's derived training projection."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sifi_streamer.capture import (
    CaptureLogReader,
    CaptureStarted,
    CaptureStopped,
    Marker,
    RawPacket,
    SegmentStarted,
    SegmentStopped,
    record_to_wire_map,
)

from qgrip.domain import ArtifactMetadata, QGripProfile, activation_target
from qgrip.errors import ArtifactError, ValidationError
from qgrip.streaming import EMG_STREAM_ID

LOGGER = logging.getLogger("qgrip.artifacts")


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
    return raw / f"capture-{stamp}.capture.jsonl.zst"


def parquet_path(capture: Path) -> Path:
    suffix = ".capture.jsonl.zst"
    if not capture.name.endswith(suffix):
        raise ArtifactError(f"not a streamer capture: {capture}")
    return capture.with_name(capture.name[: -len(suffix)] + ".parquet")


def _number(attributes: object, name: str) -> float:
    if not isinstance(attributes, dict) or not isinstance(attributes.get(name), (int, float)):
        raise ArtifactError(f"capture metadata has no numeric {name}")
    return float(attributes[name])


def _capture_header(source: Path) -> ArtifactMetadata:
    """Read the header required for export without scanning packet records."""
    records = iter(CaptureLogReader(source))
    start = next(records)
    if not isinstance(start, CaptureStarted):
        raise ArtifactError("capture does not begin with capture_started")
    attributes = start.attributes
    return ArtifactMetadata(
        source,
        "capture",
        str(attributes["subject"]),
        str(attributes["created_at"]),
        cast(Any, attributes["device"]),
        _number(attributes, "sample_rate_hz"),
        int(_number(attributes, "channels")),
        tuple(str(attributes["classes"]).split(",")),
        bool(attributes["proportional"]),
        False,
    )


def capture_metadata(path: str | Path) -> ArtifactMetadata:
    """Read metadata and verify terminal closure without retaining packets.

    Zstandard streams cannot be safely reverse-read through the public
    sifi-streamer API.  Completion is therefore verified by a streaming scan;
    export uses ``_capture_header`` and performs only its required single pass.
    """
    source = Path(path).resolve()
    try:
        header = _capture_header(source)
        records = iter(CaptureLogReader(source))
        next(records)
        complete = any(isinstance(record, CaptureStopped) for record in records)
        return replace(header, complete=complete)
    except (OSError, ValueError, KeyError, StopIteration, TypeError) as exc:
        raise ArtifactError(f"invalid capture {source}: {exc}") from exc


def iter_capture_records(path: str | Path) -> Iterator[dict[str, object]]:
    """Yield capture records lazily so large captures are never materialized."""
    source = Path(path).resolve()
    try:
        for record in CaptureLogReader(source):
            yield record_to_wire_map(record)
    except (OSError, ValueError, TypeError) as exc:
        raise ArtifactError(f"invalid capture {source}: {exc}") from exc


def read_capture(path: str | Path) -> tuple[ArtifactMetadata, Iterator[dict[str, object]]]:
    """Return capture metadata and a lazy record iterator.

    Completion is verified by streaming the capture once without retaining
    packet documents; the returned iterator opens a separate lazy read stream.
    """
    return capture_metadata(path), iter_capture_records(path)


def _emg_rows(packet: RawPacket, presentation: dict[str, object]) -> list[dict[str, object]]:
    document = packet.packet
    if document.get("packet_type") != EMG_STREAM_ID:
        return []
    timestamps, data = document.get("timestamps"), document.get("data")
    if not isinstance(timestamps, list) or not isinstance(data, dict):
        return []
    rows: list[dict[str, object]] = []
    channels = [name for name in data if isinstance(name, str) and name.startswith("emg")]
    malformed_channels = [name for name in channels if not name[3:].isdigit()]
    if malformed_channels:
        LOGGER.warning(
            "dropping EMG packet %d: invalid channel names %s; expected emg<index>",
            packet.sequence,
            ", ".join(sorted(malformed_channels)),
        )
        return rows
    channels.sort(key=lambda name: int(name[3:]))
    channels_expected = presentation.get("channels")
    if not isinstance(channels_expected, int):
        raise ArtifactError("presentation has no expected channel count")
    if len(channels) != channels_expected:
        LOGGER.warning(
            "dropping EMG packet %d: expected %d channels, received %d",
            packet.sequence,
            channels_expected,
            len(channels),
        )
        return rows
    for index, timestamp in enumerate(timestamps):
        if not isinstance(timestamp, (int, float)):
            continue
        values: list[float] = []
        for name in channels:
            channel_values = data[name]
            if not isinstance(channel_values, list) or index >= len(channel_values):
                break
            value = channel_values[index]
            if not isinstance(value, (int, float)):
                break
            values.append(float(value))
        if len(values) != channels_expected:
            LOGGER.warning(
                "dropping EMG packet %d sample %d: incomplete or nonnumeric channels",
                packet.sequence,
                index,
            )
            continue
        row: dict[str, object] = {
            "timestamp": float(timestamp),
            "gesture": presentation["gesture"],
            "activation": presentation["activation"],
            "trial": presentation["trial"],
            "sequence": packet.sequence,
            "capture_file": presentation["capture_file"],
            "device": presentation["device"],
            "sample_rate_hz": presentation["sample_rate_hz"],
            "sample_index_in_packet": index,
            "host_monotonic_ns": packet.host_monotonic_ns,
            "host_unix_ns": packet.host_unix_ns,
            "samples_lost": document.get("samples_lost", 0),
        }
        row.update({f"channel_{channel}": value for channel, value in enumerate(values)})
        rows.append(row)
    return rows


def _assign_activation_labels(
    rows: list[dict[str, object]], gesture: str, proportional: bool
) -> None:
    """Stamp each accepted sample with the activation prompt it was captured under.

    Proportional captures sweep a triangle ramp across the hold, so the label is
    reconstructed from each sample's position within the ordered presentation —
    the same shape the operator followed on screen (see ``activation_target``).
    Non-proportional captures carry a flat full contraction (``rest`` stays 0).
    """
    if not proportional:
        constant = 0.0 if gesture == "rest" else 1.0
        for row in rows:
            row["activation"] = constant
        return
    count = len(rows)
    for index, row in enumerate(rows):
        fraction = index / (count - 1) if count > 1 else 0.0
        row["activation"] = activation_target(gesture, fraction)


def _assign_activation_energy(
    rows: list[dict[str, object]], channels: int, window_samples: int
) -> None:
    """Add causal, channel-demeaned RMS energy to every presentation sample."""
    values = np.asarray(
        [
            [float(cast(float, row[f"channel_{channel}"])) for channel in range(channels)]
            for row in rows
        ],
        dtype=np.float64,
    )
    sums = np.vstack((np.zeros((1, channels)), np.cumsum(values, axis=0)))
    squared_sums = np.vstack((np.zeros((1, channels)), np.cumsum(np.square(values), axis=0)))
    for end, row in enumerate(rows, start=1):
        start = max(0, end - window_samples)
        count = end - start
        means = (sums[end] - sums[start]) / count
        mean_squares = (squared_sums[end] - squared_sums[start]) / count
        channel_variances = np.maximum(mean_squares - np.square(means), 0)
        row["activation_energy"] = float(np.sqrt(np.mean(channel_variances)))


def export_capture(path: str | Path, *, activation_energy_window_seconds: float) -> Path:
    """Project accepted QGrip SGT presentations from a generic capture log.

    ``trial``, ``presentation``, practice/calibration stages, activation, and
    repetition are QGrip semantics.  They deliberately live here rather than in
    sifi-streamer, whose job is to preserve the packets and generic boundaries.
    """
    source = Path(path).resolve()
    if not math.isfinite(activation_energy_window_seconds) or activation_energy_window_seconds <= 0:
        raise ArtifactError("activation energy window must be finite and positive")
    try:
        metadata = _capture_header(source)
    except (OSError, ValueError, KeyError, StopIteration, TypeError) as exc:
        raise ArtifactError(f"invalid capture {source}: {exc}") from exc
    output = parquet_path(metadata.path)
    if output.exists():
        raise ArtifactError(f"derived export already exists: {output}")
    presentations: dict[str, dict[str, object]] = {}
    active_presentation: str | None = None
    complete = False
    for record in CaptureLogReader(metadata.path):
        if isinstance(record, SegmentStarted) and record.segment_kind == "presentation":
            attrs = record.attributes
            label, trial, activation = (
                attrs.get("label"),
                attrs.get("trial"),
                attrs.get("activation"),
            )
            if (
                isinstance(label, str)
                and isinstance(trial, int)
                and isinstance(activation, (int, float))
            ):
                presentations[record.segment_id] = {
                    "gesture": label,
                    "trial": trial,
                    "activation": float(activation),
                    "channels": metadata.channels,
                    "capture_file": str(metadata.path),
                    "device": metadata.device,
                    "sample_rate_hz": metadata.sample_rate_hz,
                    "rows": [],
                    "stop_reason": None,
                    "superseded": False,
                }
                active_presentation = record.segment_id
        elif isinstance(record, Marker):
            presentation = presentations.get(record.marker_id)
            if record.marker_kind == "presentation_superseded" and presentation is not None:
                presentation["superseded"] = True
            elif record.marker_kind == "activation_target":
                target = presentations.get(active_presentation or "")
                value = record.attributes.get("activation")
                if target is not None and isinstance(value, (int, float)):
                    target["activation"] = float(value)
        elif isinstance(record, RawPacket) and active_presentation is not None:
            presentation = presentations[active_presentation]
            cast(list[dict[str, object]], presentation["rows"]).extend(
                _emg_rows(record, presentation)
            )
        elif isinstance(record, SegmentStopped):
            presentation = presentations.get(record.segment_id)
            if presentation is not None:
                presentation["stop_reason"] = record.reason
                if active_presentation == record.segment_id:
                    active_presentation = None
        elif isinstance(record, CaptureStopped):
            complete = True
    if not complete:
        raise ArtifactError(f"capture did not close cleanly: {metadata.path}")
    accepted: list[dict[str, object]] = []
    energy_window_samples = max(
        1, round(metadata.sample_rate_hz * activation_energy_window_seconds)
    )
    for presentation in presentations.values():
        if presentation["stop_reason"] != "completed" or presentation["superseded"]:
            continue
        rows = cast(list[dict[str, object]], presentation["rows"])
        _assign_activation_labels(rows, str(presentation["gesture"]), metadata.proportional)
        _assign_activation_energy(rows, metadata.channels, energy_window_samples)
        accepted.extend(rows)
    if not accepted:
        raise ArtifactError(f"capture has no accepted EMG rows: {metadata.path}")
    temporary = output.with_suffix(".parquet.tmp")
    table = pa.Table.from_pylist(accepted)
    table = table.replace_schema_metadata(
        {
            **(table.schema.metadata or {}),
            b"qgrip.activation_energy.method": b"causal_rms",
            b"qgrip.activation_energy.window_seconds": str(
                activation_energy_window_seconds
            ).encode(),
            b"qgrip.activation_energy.window_samples": str(energy_window_samples).encode(),
        }
    )
    pq.write_table(table, temporary)
    temporary.replace(output)
    return output


def discover_artifacts(profile: QGripProfile, subject: str | None = None) -> tuple[Path, ...]:
    scope = profile.data_root / subject if subject else profile.data_root
    if not scope.exists():
        return ()
    found = [
        path.resolve()
        for pattern in ("*.capture.jsonl.zst", "*.parquet", "*.pt", "*.onnx")
        for path in scope.rglob(pattern)
    ]
    return tuple(sorted(set(found), key=lambda item: item.stat().st_mtime, reverse=True))


def latest_capture(profile: QGripProfile, subject: str) -> Path:
    captures = [
        path
        for path in discover_artifacts(profile, subject)
        if path.name.endswith(".capture.jsonl.zst")
    ]
    if not captures:
        raise ArtifactError(f"no completed capture found for {subject}")
    return captures[0]
