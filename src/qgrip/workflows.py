"""Framework-independent SGT, training, inference, and lifecycle coordination."""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sifi_streamer.acquisition import create_capture_runtime

from qgrip.artifacts import (
    export_capture,
    new_capture_path,
)
from qgrip.domain import (
    EpochMetric,
    JobState,
    JobStatus,
    Prediction,
    QGripProfile,
    SGTCommand,
    SGTProgress,
    SGTRequest,
    TrainingRequest,
    activation_target,
)
from qgrip.errors import ArtifactError, BusyError
from qgrip.streaming import (
    LiveEMGSession,
    PredictionDebouncer,
    sample_rates_match,
    streamer_config,
    streamer_device_factory,
)

ProgressCallback = Callable[[SGTProgress], None]


class SGTCommandGate:
    """Thread-safe control channel for interactive Screen Guided Training.

    The capture thread blocks on :meth:`take` at gate points (after each recorded
    presentation in manual mode, or whenever paused); adapters push proceed,
    repeat, pause, resume, or abort commands through :meth:`send`.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._pending: deque[SGTCommand] = deque()

    def send(self, command: SGTCommand) -> None:
        with self._condition:
            self._pending.append(command)
            self._condition.notify_all()

    def drain(self) -> list[SGTCommand]:
        """Return and clear queued commands without blocking."""
        with self._condition:
            queued = list(self._pending)
            self._pending.clear()
            return queued

    def take(self, cancel: threading.Event, poll_seconds: float = 0.1) -> SGTCommand | None:
        """Block until a command arrives; return ``None`` when cancelled."""
        with self._condition:
            while not self._pending:
                if cancel.is_set():
                    return None
                self._condition.wait(poll_seconds)
            return self._pending.popleft()


class SGTService:
    def run(
        self,
        request: SGTRequest,
        cancel: threading.Event,
        progress: ProgressCallback | None = None,
        gate: SGTCommandGate | None = None,
    ) -> Path:
        profile = request.profile
        output = new_capture_path(profile, request.subject)
        total = len(profile.sgt.gestures) * profile.sgt.trials
        runtime = create_capture_runtime(
            output,
            output.stem,
            streamer_device_factory(profile.device),
            {
                "experiment": "screen_guided_training",
                "subject": request.subject,
                "created_at": datetime.now(UTC).isoformat(),
                "device": profile.device.kind,
                "sample_rate_hz": profile.device.sample_rate_hz,
                "channels": profile.device.channels,
                "classes": ",".join(profile.sgt.gestures),
                "proportional": request.proportional,
            },
            config=streamer_config(profile.acquisition),
        )
        reason = "aborted"
        try:
            with runtime.controller:
                segment_sequence = 0
                recorded_presentation = 0
                paused = False

                def prompt_at(gesture: str, elapsed: float) -> float:
                    """Activation the operator is asked to hold at ``elapsed`` seconds.

                    Proportional capture sweeps a triangle ramp so a single hold
                    covers the whole activation range; the identical shape is
                    reconstructed per sample during export to label training.
                    Non-proportional capture holds a flat full contraction.
                    """
                    if not request.proportional:
                        return 0.0 if gesture == "rest" else 1.0
                    duration = profile.sgt.duration_seconds
                    fraction = elapsed / duration if duration > 0 else 0.0
                    return activation_target(gesture, fraction)

                def run_preparation(gesture: str) -> None:
                    """Give the operator a brief get-ready countdown before a prompt.

                    The preparation period paces the transition into practice
                    and recorded presentations so the operator can settle
                    before each stimulus.  It records nothing: no capture segment is
                    opened, only progress updates are emitted.  A non-positive
                    ``preparation_seconds`` disables it entirely.
                    """
                    seconds = profile.sgt.preparation_seconds
                    if seconds <= 0:
                        return
                    started = time.monotonic()
                    while (elapsed := time.monotonic() - started) < seconds:
                        if progress:
                            progress(
                                SGTProgress(
                                    JobState.RUNNING,
                                    gesture=gesture,
                                    stage="preparation",
                                    instruction=_stage_instruction("preparation", gesture),
                                    stimulus_image=_stimulus_image(profile, gesture),
                                    total_trials=total,
                                    elapsed_seconds=elapsed,
                                    duration_seconds=seconds,
                                    capture=output,
                                )
                            )
                        if cancel.wait(
                            min(profile.sgt.progress_interval_seconds, seconds - elapsed)
                        ):
                            raise InterruptedError("capture cancelled")

                def run_unrecorded_stage(kind: str, gesture: str) -> None:
                    """Capture practice evidence without training labels.

                    The practice stage mirrors the recorded presentation prompt so
                    the operator can rehearse following the activation ramp; the
                    same triangle target is surfaced via ``activation`` but nothing
                    here is retained as a training label.
                    """
                    nonlocal segment_sequence
                    segment_sequence += 1
                    stage_id = f"{kind}-{segment_sequence:03d}"
                    if progress:
                        progress(
                            SGTProgress(
                                JobState.RUNNING,
                                gesture=gesture,
                                stage=kind,
                                instruction=_stage_instruction(kind, gesture),
                                stimulus_image=_stimulus_image(profile, gesture),
                                total_trials=total,
                                duration_seconds=profile.sgt.duration_seconds,
                                activation=prompt_at(gesture, 0.0),
                                capture=output,
                            )
                        )
                    runtime.controller.start_segment(stage_id, kind, label=gesture)
                    runtime.controller.marker(stage_id, f"{kind}_started", label=gesture)
                    started = time.monotonic()
                    try:
                        while (
                            elapsed := time.monotonic() - started
                        ) < profile.sgt.duration_seconds:
                            if cancel.wait(
                                min(
                                    profile.sgt.progress_interval_seconds,
                                    profile.sgt.duration_seconds - elapsed,
                                )
                            ):
                                raise InterruptedError("capture cancelled")
                            if progress:
                                progress(
                                    SGTProgress(
                                        JobState.RUNNING,
                                        gesture=gesture,
                                        stage=kind,
                                        instruction=_stage_instruction(kind, gesture),
                                        stimulus_image=_stimulus_image(profile, gesture),
                                        total_trials=total,
                                        elapsed_seconds=elapsed,
                                        duration_seconds=profile.sgt.duration_seconds,
                                        activation=prompt_at(gesture, elapsed),
                                        capture=output,
                                    )
                                )
                    finally:
                        runtime.controller.stop_segment(
                            stage_id, "completed" if not cancel.is_set() else "aborted"
                        )

                if profile.sgt.practice:
                    for gesture in profile.sgt.gestures:
                        run_preparation(gesture)
                        run_unrecorded_stage("practice", gesture)

                def emit_presentation(
                    gesture: str,
                    trial_index: int,
                    activation: float,
                    elapsed: float,
                    *,
                    awaiting: bool = False,
                ) -> None:
                    if not progress:
                        return
                    progress(
                        SGTProgress(
                            JobState.RUNNING,
                            gesture=gesture,
                            stage="awaiting" if awaiting else "presentation",
                            instruction=_stage_instruction("presentation", gesture),
                            stimulus_image=_stimulus_image(profile, gesture),
                            trial=trial_index,
                            total_trials=total,
                            elapsed_seconds=elapsed,
                            duration_seconds=profile.sgt.duration_seconds,
                            activation=activation,
                            capture=output,
                            awaiting_command=awaiting,
                        )
                    )

                def await_gate(gesture: str, trial_index: int, activation: float) -> str:
                    """Block until the operator proceeds or repeats the stimulus.

                    Auto mode advances immediately unless paused; manual mode always
                    waits.  Returns ``"repeat"`` to redo the stimulus, else ``"next"``.
                    """
                    nonlocal paused
                    if request.auto and not paused:
                        for command in gate.drain() if gate else []:
                            if command == SGTCommand.PAUSE:
                                paused = True
                            elif command == SGTCommand.RESUME:
                                paused = False
                            elif command == SGTCommand.REPEAT:
                                return "repeat"
                            elif command == SGTCommand.ABORT:
                                cancel.set()
                                raise InterruptedError("capture cancelled")
                        if not paused:
                            return "next"
                    if gate is None:
                        return "next"
                    emit_presentation(
                        gesture,
                        trial_index,
                        activation,
                        profile.sgt.duration_seconds,
                        awaiting=True,
                    )
                    while True:
                        command = gate.take(cancel)
                        if command is None or command == SGTCommand.ABORT:
                            cancel.set()
                            raise InterruptedError("capture cancelled")
                        if command == SGTCommand.REPEAT:
                            return "repeat"
                        if command == SGTCommand.RESUME:
                            paused = False
                            return "next"
                        if command == SGTCommand.PAUSE:
                            paused = True

                def present_gesture(gesture: str, trial: int) -> None:
                    nonlocal segment_sequence, recorded_presentation
                    duration = profile.sgt.duration_seconds
                    peak = prompt_at(gesture, duration / 2)
                    while True:
                        if cancel.is_set():
                            raise InterruptedError("capture cancelled")
                        run_preparation(gesture)
                        segment_sequence += 1
                        trial_index = recorded_presentation + 1
                        emit_presentation(gesture, trial_index, prompt_at(gesture, 0.0), 0.0)
                        presentation_id = runtime.controller.start_segment(
                            f"presentation-{trial:03d}-{segment_sequence:03d}",
                            "presentation",
                            label=gesture,
                            trial=trial,
                            activation=peak,
                        )
                        runtime.controller.marker(
                            presentation_id,
                            "activation_target",
                            label=gesture,
                            activation=peak,
                        )
                        started = time.monotonic()
                        while (elapsed := time.monotonic() - started) < duration:
                            if cancel.wait(
                                min(
                                    profile.sgt.progress_interval_seconds,
                                    duration - elapsed,
                                )
                            ):
                                raise InterruptedError("capture cancelled")
                            emit_presentation(
                                gesture, trial_index, prompt_at(gesture, elapsed), elapsed
                            )
                        runtime.controller.stop_segment(presentation_id, "completed")
                        if await_gate(gesture, trial_index, peak) == "repeat":
                            runtime.controller.marker(
                                presentation_id, "presentation_superseded", label=gesture
                            )
                            continue
                        recorded_presentation = trial_index
                        return

                for trial in range(1, profile.sgt.trials + 1):
                    trial_id = runtime.controller.start_segment(
                        f"trial-{trial:03d}", "trial", trial=trial
                    )
                    try:
                        for gesture in profile.sgt.gestures:
                            present_gesture(gesture, trial)
                    finally:
                        runtime.controller.stop_segment(
                            trial_id, "completed" if not cancel.is_set() else "aborted"
                        )
            reason = "normal_completion"
        except InterruptedError:
            raise
        finally:
            if runtime.controller.started:
                runtime.controller.close(reason)
        if progress:
            progress(SGTProgress(JobState.COMPLETED, total_trials=total, capture=output))
        return output


def _stage_instruction(stage: str, gesture: str) -> str:
    label = gesture.replace("_", " ")
    if stage == "preparation":
        return f"Get ready: {label}."
    if stage == "practice":
        return f"Practice: {label}."
    return f"Perform: {label}."


def _stimulus_image(profile: QGripProfile, gesture: str) -> str | None:
    """Return a verified local stimulus filename without exposing filesystem paths."""
    filename = f"{gesture}.png"
    if Path(filename).name != filename:
        return None
    return filename if (profile.assets_root / filename).is_file() else None


class TrainingService:
    """Lazy facade keeping Torch optional until training is requested."""

    PRESET_VERSION = 1

    def train(
        self,
        request: TrainingRequest,
        cancel: threading.Event,
        metric: Callable[[EpochMetric], None] | None = None,
    ) -> Path:
        try:
            from qgrip.training import TorchTrainingService
        except ImportError as exc:
            raise ArtifactError(
                "training requires the qgrip train extra: uv sync --extra train"
            ) from exc
        return TorchTrainingService(request.profile.training).train(request, cancel, metric)


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
        # A Condition (rather than a plain Lock) lets SSE clients block until the
        # status actually changes instead of polling on a fixed interval.
        self._lock = threading.Condition()
        self._status: JobStatus | None = None
        self._version = 0
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._sgt_gate: SGTCommandGate | None = None

    @property
    def status(self) -> JobStatus | None:
        with self._lock:
            return self._status

    def _notify(self) -> None:
        """Bump the status version and wake waiters. Caller must hold ``_lock``."""
        self._version += 1
        self._lock.notify_all()

    def status_snapshot(self) -> tuple[JobStatus | None, int]:
        """Return the current status together with its monotonic version."""
        with self._lock:
            return self._status, self._version

    def status_since(self, version: int, timeout: float) -> tuple[JobStatus | None, int]:
        """Block until the status advances past ``version`` (or ``timeout`` elapses).

        Returns the latest status and version so callers push updates only when
        something changed, replacing busy polling of :attr:`status`.
        """
        with self._lock:
            self._lock.wait_for(lambda: self._version != version, timeout)
            return self._status, self._version

    def _begin(self, kind: str, target: Callable[[], str]) -> JobStatus:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise BusyError("another hardware-owning operation is active")
            status = JobStatus(uuid.uuid4().hex, kind, JobState.RUNNING)
            self._status = status
            self._notify()
            self._cancel.clear()

            def runner() -> None:
                try:
                    result = target()
                    state = JobState.CANCELLED if self._cancel.is_set() else JobState.COMPLETED
                    final = replace(status, state=state, progress=1.0, result=result)
                except InterruptedError as exc:
                    final = replace(status, state=JobState.CANCELLED, message=str(exc))
                except Exception as exc:
                    final = replace(status, state=JobState.FAILED, message=str(exc))
                with self._lock:
                    self._status = final
                    self._notify()

            self._thread = threading.Thread(target=runner, name=f"qgrip-{kind}", daemon=False)
            self._thread.start()
            return status

    def start_sgt(self, request: SGTRequest) -> JobStatus:
        gate = SGTCommandGate()
        self._sgt_gate = gate

        def update(value: SGTProgress) -> None:
            with self._lock:
                if self._status:
                    current = value.trial / max(1, value.total_trials)
                    self._status = replace(
                        self._status,
                        progress=min(0.99, current),
                        message=value.instruction or value.gesture or "",
                        gesture=value.gesture,
                        stage=value.stage,
                        instruction=value.instruction,
                        stimulus_image=value.stimulus_image,
                        elapsed_seconds=value.elapsed_seconds,
                        duration_seconds=value.duration_seconds,
                        activation=value.activation,
                        awaiting_command=value.awaiting_command,
                    )
                    self._notify()

        return self._begin("sgt", lambda: str(self.sgt.run(request, self._cancel, update, gate)))

    def send_sgt_command(self, command: SGTCommand) -> None:
        """Forward an interactive Screen Guided Training control command."""
        if command == SGTCommand.ABORT:
            self.cancel()
        gate = self._sgt_gate
        if gate is not None:
            gate.send(command)

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
                        progress=value.epoch / request.profile.training.epochs,
                        message=f"epoch {value.epoch}/{request.profile.training.epochs}",
                        metrics=tuple(metrics),
                    )
                    self._notify()

        return self._begin(
            "training", lambda: str(self.training.train(request, self._cancel, update))
        )

    def start_inference(self, model: Path, profile: QGripProfile) -> JobStatus:
        inference = InferenceService(model, profile.inference.backend)

        def run() -> str:
            with LiveEMGSession(profile.device, profile.acquisition) as session:
                if session.channels != inference.channels:
                    raise ArtifactError("model channel count does not match live EMG stream")
                if not sample_rates_match(
                    session.sample_rate_hz, float(inference.metadata["sample_rate_hz"])
                ):
                    raise ArtifactError("model sample rate does not match live EMG stream")
                minimum_new_samples = max(
                    1,
                    round(session.sample_rate_hz * profile.inference.inference_period_seconds),
                )
                debouncer = PredictionDebouncer(profile.inference.switch_predictions)
                next_inference_at = time.monotonic()
                while not self._cancel.is_set():
                    wait_seconds = next_inference_at - time.monotonic()
                    if wait_seconds > 0:
                        self._cancel.wait(min(wait_seconds, profile.inference.maximum_wait_seconds))
                        continue
                    samples = session.next_window(inference.window_size, minimum_new_samples)
                    health = session.health
                    if samples is None:
                        with self._lock:
                            if self._status:
                                self._status = replace(self._status, health=health)
                                self._notify()
                        self._cancel.wait(profile.inference.idle_poll_seconds)
                        continue
                    prediction = inference.predict(samples)
                    if prediction.confidence < profile.inference.confidence_gate:
                        prediction = replace(prediction, gesture="rest")
                    accepted = debouncer.accept(prediction)
                    if accepted is not None:
                        with self._lock:
                            if self._status:
                                self._status = replace(
                                    self._status,
                                    message=accepted.gesture,
                                    prediction=accepted,
                                    health=health,
                                )
                                self._notify()
                    next_inference_at += profile.inference.inference_period_seconds
                    if next_inference_at < time.monotonic():
                        next_inference_at = time.monotonic()
                return str(model)

        return self._begin("inference", run)

    def cancel(self) -> None:
        self._cancel.set()

    def close(self) -> None:
        self.cancel()
        with self._lock:
            # Wake any SSE waiters so they can observe the shutdown promptly.
            self._notify()
        if self._thread is not None:
            self._thread.join(timeout=10)
