"""Framework-independent SGT, training, inference, and lifecycle coordination."""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from qgrip.artifacts import (
    export_capture,
    new_capture_path,
    write_capture_header,
)
from qgrip.devices import SignalDevice, create_device
from qgrip.domain import (
    ArtifactMetadata,
    DeviceConfig,
    EpochMetric,
    JobStatus,
    Prediction,
    QGripProfile,
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
                        device.set_cue(gesture, activation)
                        if progress:
                            progress(
                                SGTProgress(
                                    "running",
                                    gesture,
                                    trial_number,
                                    total,
                                    0,
                                    activation,
                                    output,
                                )
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
                                        trial_number,
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
    """Lazy facade keeping Torch optional until training is requested."""

    PRESET_VERSION = 1

    def __init__(
        self,
        *,
        epochs: int = 30,
        batch_size: int = 128,
        window_seconds: float = 1.0,
        stride_seconds: float = 0.05,
        export_onnx: bool = True,
    ) -> None:
        self._epochs = epochs
        self._batch_size = batch_size
        self._window_seconds = window_seconds
        self._stride_seconds = stride_seconds
        self._export_onnx = export_onnx

    @property
    def epochs(self) -> int:
        return self._epochs

    def train(
        self,
        request: TrainingRequest,
        cancel: threading.Event,
        metric: Callable[[EpochMetric], None] | None = None,
    ) -> Path:
        try:
            from qgrip.training import TorchTrainingService, TrainingOptions
        except ImportError as exc:
            raise ArtifactError(
                "training requires the qgrip train extra: uv sync --extra train"
            ) from exc
        return TorchTrainingService(
            TrainingOptions(
                epochs=self._epochs,
                batch_size=self._batch_size,
                window_seconds=self._window_seconds,
                stride_seconds=self._stride_seconds,
                export_onnx=self._export_onnx,
            )
        ).train(request, cancel, metric)


class InferenceService:
    """Stateful streaming inference using a self-describing Torch or ONNX model."""

    def __init__(self, model: str | Path, backend: str = "auto") -> None:
        self.path = Path(model).resolve()
        try:
            import torch

            from qgrip.models import ONNXEMGClassifier, load_checkpoint, load_model_checkpoint
        except ImportError as exc:
            raise ArtifactError(
                "inference requires the qgrip train extra: uv sync --extra train"
            ) from exc
        self._torch = torch
        self._torch_model: Any | None = None
        self._onnx_model: ONNXEMGClassifier | None = None
        checkpoint_path = self.path
        if self.path.suffix == ".onnx":
            checkpoint_path = self.path.with_suffix(".pt")
            backend = "onnx"
        try:
            self.metadata = load_checkpoint(checkpoint_path)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            raise ArtifactError(f"unsupported checkpoint {checkpoint_path}: {exc}") from exc
        requested_backend = backend
        onnx_path = checkpoint_path.with_suffix(".onnx")
        automatic = requested_backend == "auto"
        if requested_backend == "auto":
            requested_backend = "onnx" if onnx_path.exists() else "torch"
        if requested_backend == "onnx":
            if not onnx_path.exists():
                raise ArtifactError(f"ONNX model not found beside checkpoint: {onnx_path}")
            try:
                self._onnx_model = ONNXEMGClassifier(onnx_path, prefer_cuda=True)
            except (ImportError, OSError, RuntimeError) as exc:
                if not automatic:
                    raise ArtifactError(f"cannot load ONNX model {onnx_path}: {exc}") from exc
                requested_backend = "torch"
        if requested_backend == "torch":
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._torch_model, self.metadata = load_model_checkpoint(checkpoint_path, torch_device)
            self._torch_device = torch_device
        elif requested_backend != "onnx":
            raise ArtifactError(f"unsupported inference backend: {backend}")
        self.backend = requested_backend
        self.window_size = int(self.metadata["window_size"])
        self.channels = int(self.metadata.get("n_channels", self.metadata.get("channels", 0)))
        self.labels = tuple(str(value) for value in self.metadata["labels"])
        self._samples: deque[tuple[float, ...]] = deque(maxlen=self.window_size)

    def predict(self, samples: tuple[tuple[float, ...], ...]) -> Prediction:
        started = time.perf_counter()
        for sample in samples:
            if len(sample) != self.channels:
                raise ArtifactError(
                    f"model expects {self.channels} channels, received {len(sample)}"
                )
            self._samples.append(sample)
        window = np.zeros((self.window_size, self.channels), dtype=np.float32)
        buffered = np.asarray(self._samples, dtype=np.float32)
        if len(buffered):
            window[-len(buffered) :] = buffered
        if self._onnx_model is not None:
            logits, activation_output = self._onnx_model.predict(window[None, ...])
            scores = np.asarray(logits[0], dtype=float)
        else:
            if self._torch_model is None:
                raise ArtifactError("Torch model is not loaded")
            tensor = self._torch.from_numpy(window).unsqueeze(0).to(self._torch_device)
            with self._torch.inference_mode():
                output = self._torch_model(tensor)
            if isinstance(output, tuple):
                logits_tensor, activation_tensor = output
                activation_output = activation_tensor.detach().cpu().numpy()
            else:
                logits_tensor = output
                activation_output = None
            scores = logits_tensor.detach().cpu().numpy()[0]
        shifted = scores - scores.max()
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        index = int(probabilities.argmax())
        gesture = self.labels[index]
        confidence = float(probabilities[index])
        activation = 1.0
        if activation_output is not None:
            activation = float(np.clip(np.asarray(activation_output).reshape(-1)[0], 0, 1))
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
                        self._status,
                        progress=value.epoch / self.training.epochs,
                        message=f"epoch {value.epoch}/{self.training.epochs}",
                        metrics=tuple(metrics),
                    )

        return self._begin(
            "training", lambda: str(self.training.train(request, self._cancel, update))
        )

    def start_inference(self, model: Path, profile: QGripProfile) -> JobStatus:
        inference = InferenceService(model, profile.inference.backend)

        def run() -> str:
            device = create_device(profile.device)
            try:
                device.connect()
                while not self._cancel.is_set():
                    packet = device.read(
                        max(1, int(device.sample_rate_hz * profile.inference.interval_seconds))
                    )
                    prediction = inference.predict(packet.samples)
                    if prediction.confidence < profile.inference.confidence_gate:
                        prediction = replace(prediction, gesture="rest")
                    with self._lock:
                        if self._status:
                            self._status = replace(
                                self._status,
                                message=prediction.gesture,
                                prediction=prediction,
                            )
                    if profile.device.kind == "synthetic":
                        time.sleep(profile.inference.interval_seconds)
                return str(model)
            finally:
                device.close()

        return self._begin("inference", run)

    def cancel(self) -> None:
        self._cancel.set()

    def close(self) -> None:
        self.cancel()
        if self._thread is not None:
            self._thread.join(timeout=10)
