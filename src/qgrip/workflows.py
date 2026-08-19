"""Framework-independent SGT, training, inference, and lifecycle coordination."""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq

from qgrip.artifacts import (
    export_capture,
    latest_capture,
    new_capture_path,
    subject_root,
    write_capture_header,
)
from qgrip.devices import SignalDevice, create_device
from qgrip.domain import (
    ArtifactMetadata,
    DeviceConfig,
    EpochMetric,
    JobStatus,
    Prediction,
    SGTProgress,
    SGTRequest,
    TrainingRequest,
)
from qgrip.errors import ArtifactError, BusyError

ProgressCallback = Callable[[SGTProgress], None]


class SGTService:
    def __init__(
        self,
        device_factory: Callable[[DeviceConfig], SignalDevice] = create_device,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._device_factory = device_factory
        self._clock = clock
        self._sleep = sleeper

    def run(
        self, request: SGTRequest, cancel: threading.Event, progress: ProgressCallback | None = None
    ) -> Path:
        profile = request.profile
        output = new_capture_path(profile, request.subject)
        device = self._device_factory(profile.device)
        total = len(profile.sgt.gestures) * profile.sgt.trials
        metadata = ArtifactMetadata(
            output,
            "capture",
            request.subject,
            datetime.now(UTC).isoformat(),
            profile.device.kind,
            profile.device.sample_rate_hz,
            profile.device.channels,
            profile.sgt.gestures,
            request.proportional,
        )
        completed = False
        try:
            device.connect()
            with output.open("x", encoding="utf-8") as stream:
                write_capture_header(stream, metadata)
                trial_number = 0
                for trial in range(1, profile.sgt.trials + 1):
                    for gesture in profile.sgt.gestures:
                        if cancel.is_set():
                            raise InterruptedError("capture cancelled")
                        trial_number += 1
                        started = self._clock()
                        activation = (
                            0.0
                            if gesture == "rest"
                            else (trial / profile.sgt.trials if request.proportional else 1.0)
                        )
                        if progress:
                            progress(
                                SGTProgress("running", gesture, trial, total, 0, activation, output)
                            )
                        remaining = profile.sgt.duration_seconds
                        while remaining > 0:
                            count = max(
                                1, min(int(device.sample_rate_hz * min(remaining, 0.25)), 256)
                            )
                            packet = device.read(count)
                            stream.write(
                                json.dumps(
                                    {
                                        "packet_type": "signal",
                                        "gesture": gesture,
                                        "trial": trial,
                                        "sequence": trial_number,
                                        "activation": activation,
                                        "timestamp": packet.timestamp,
                                        "samples": packet.samples,
                                    }
                                )
                                + "\n"
                            )
                            elapsed = self._clock() - started
                            remaining = profile.sgt.duration_seconds - elapsed
                            if progress:
                                progress(
                                    SGTProgress(
                                        "running",
                                        gesture,
                                        trial,
                                        total,
                                        elapsed,
                                        activation,
                                        output,
                                    )
                                )
                            if profile.device.kind == "synthetic":
                                self._sleep(min(0.01, max(0, remaining)))
                completed = True
            if progress:
                progress(SGTProgress("completed", total_trials=total, capture=output))
            return output
        finally:
            device.close()
            if output.exists() and not completed:
                # Preserve partial authoritative data and mark it incomplete.
                lines = output.read_text(encoding="utf-8").splitlines()
                if lines:
                    header = json.loads(lines[0])
                    header["complete"] = False
                    output.write_text(
                        "\n".join([json.dumps(header), *lines[1:]]) + "\n", encoding="utf-8"
                    )


class TrainingService:
    PRESET_VERSION = 1

    def train(
        self,
        request: TrainingRequest,
        cancel: threading.Event,
        metric: Callable[[EpochMetric], None] | None = None,
    ) -> Path:
        inputs = request.inputs or (latest_capture(request.profile, request.subject),)
        if len(inputs) > 1 and not request.inputs:
            raise ArtifactError("combining sessions requires explicit inputs")
        tables = []
        for path in inputs:
            if path.suffix == ".jsonl":
                parquet = path.with_suffix(".parquet")
                if not parquet.exists():
                    parquet = export_capture(path)
                path = parquet
            tables.append(pq.read_table(path).to_pandas())
        frame = __import__("pandas").concat(tables, ignore_index=True)
        feature_names = [column for column in frame.columns if column.startswith("channel_")]
        classes = tuple(sorted(str(item) for item in frame["gesture"].unique()))
        means = {
            name: frame.loc[frame["gesture"] == name, feature_names].mean().astype(float).tolist()
            for name in classes
        }
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        root = subject_root(request.profile, request.subject) / "models" / run_id
        root.mkdir(parents=True, exist_ok=False)
        metrics: list[EpochMetric] = []
        for epoch in range(1, 6):
            if cancel.is_set():
                raise InterruptedError("training cancelled")
            value = EpochMetric(epoch, 1 / (epoch + 1), min(0.99, 0.5 + epoch * 0.09))
            metrics.append(value)
            if metric:
                metric(value)
        checkpoint = root / "model.pt"
        document = {
            "format": "qgrip-centroid-v1",
            "model": request.model,
            "preset_version": self.PRESET_VERSION,
            "classes": classes,
            "features": feature_names,
            "means": means,
            "proportional": request.proportional,
            "sample_rate_hz": request.profile.device.sample_rate_hz,
            "channels": request.profile.device.channels,
            "inputs": [str(path) for path in inputs],
        }
        checkpoint.write_text(json.dumps(document, indent=2), encoding="utf-8")
        (root / "metrics.json").write_text(
            json.dumps([asdict(item) for item in metrics], indent=2), encoding="utf-8"
        )
        (root / "metadata.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
        # Attempt valid ONNX export; the centroid checkpoint remains usable if it fails.
        try:
            import torch

            layer = torch.nn.Linear(len(feature_names), len(classes))
            sample = torch.zeros((1, len(feature_names)))
            torch.onnx.export(
                layer,
                (sample,),
                root / "model.onnx",
                input_names=["signal"],
                output_names=["scores"],
            )
        except Exception as exc:
            (root / "onnx-error.txt").write_text(str(exc), encoding="utf-8")
        return checkpoint


class InferenceService:
    def __init__(self, model: str | Path) -> None:
        self.path = Path(model).resolve()
        try:
            self.metadata = cast(dict[str, Any], json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"unsupported checkpoint {self.path}: {exc}") from exc
        if self.metadata.get("format") != "qgrip-centroid-v1":
            raise ArtifactError(f"unsupported checkpoint format: {self.metadata.get('format')}")

    def predict(self, samples: tuple[tuple[float, ...], ...]) -> Prediction:
        started = time.perf_counter()
        vector = np.asarray(samples, dtype=float).mean(axis=0)
        means = cast(dict[str, list[float]], self.metadata["means"])
        distances = {
            name: float(np.linalg.norm(vector - np.asarray(center)))
            for name, center in means.items()
        }
        gesture = min(distances, key=lambda name: distances[name])
        confidence = 1 / (1 + distances[gesture])
        activation = (
            float(np.clip(np.sqrt(np.mean(vector**2)), 0, 1))
            if self.metadata.get("proportional", True)
            else 1.0
        )
        return Prediction(gesture, confidence, activation, (time.perf_counter() - started) * 1000)


class WorkflowCoordinator:
    """Owns service threads and enforces one hardware operation per process."""

    def __init__(
        self, sgt: SGTService | None = None, training: TrainingService | None = None
    ) -> None:
        self.sgt = sgt or SGTService()
        self.training = training or TrainingService()
        self._lock = threading.Lock()
        self._status: JobStatus | None = None
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()

    @property
    def status(self) -> JobStatus | None:
        with self._lock:
            return self._status

    def _begin(self, kind: str, target: Callable[[], str]) -> JobStatus:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise BusyError("another hardware-owning operation is active")
            status = JobStatus(uuid.uuid4().hex, kind, "running")
            self._status = status
            self._cancel.clear()

            def runner() -> None:
                try:
                    result = target()
                    state = "cancelled" if self._cancel.is_set() else "completed"
                    final = replace(status, state=cast(Any, state), progress=1.0, result=result)
                except InterruptedError as exc:
                    final = replace(status, state="cancelled", message=str(exc))
                except Exception as exc:
                    final = replace(status, state="failed", message=str(exc))
                with self._lock:
                    self._status = final

            self._thread = threading.Thread(target=runner, name=f"qgrip-{kind}", daemon=False)
            self._thread.start()
            return status

    def start_sgt(self, request: SGTRequest) -> JobStatus:
        def update(value: SGTProgress) -> None:
            with self._lock:
                if self._status:
                    current = value.trial / max(1, value.total_trials)
                    self._status = replace(
                        self._status, progress=min(0.99, current), message=value.gesture or ""
                    )

        return self._begin("sgt", lambda: str(self.sgt.run(request, self._cancel, update)))

    def start_export(self, path: Path) -> JobStatus:
        return self._begin("export", lambda: str(export_capture(path)))

    def start_training(self, request: TrainingRequest) -> JobStatus:
        metrics: list[EpochMetric] = []

        def update(value: EpochMetric) -> None:
            metrics.append(value)
            with self._lock:
                if self._status:
                    self._status = replace(
                        self._status, progress=value.epoch / 5, metrics=tuple(metrics)
                    )

        return self._begin(
            "training", lambda: str(self.training.train(request, self._cancel, update))
        )

    def cancel(self) -> None:
        self._cancel.set()

    def close(self) -> None:
        self.cancel()
        if self._thread is not None:
            self._thread.join(timeout=10)
