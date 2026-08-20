"""Self-contained EMG windowing and Torch classifier training."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Sized
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from qgrip.artifacts import export_capture, latest_capture, parquet_path, subject_root
from qgrip.domain import (
    ClassSampleCount,
    EpochMetric,
    TrainingConfig,
    TrainingRequest,
    TrainingSummary,
)
from qgrip.errors import ArtifactError
from qgrip.models import CHECKPOINT_VERSION, BaseEMGClassifier, create_model, export_model_to_onnx

LOGGER = logging.getLogger("qgrip.training")


@dataclass(frozen=True, slots=True)
class ActivationCalibration:
    """Training-only energy references used to derive proportional targets."""

    method: str
    window_samples: int
    rest_floor: float
    reference_quantile: float
    class_references: tuple[tuple[str, float], ...]


def _fit_activation_calibration(
    labels: tuple[str, ...],
    targets: list[int],
    energies: list[float],
    train_indices: list[int],
    *,
    window_samples: int,
    reference_quantile: float,
) -> tuple[ActivationCalibration, list[float]]:
    """Fit energy scaling on training indices, then transform the complete dataset."""
    if "rest" not in labels:
        raise ArtifactError("proportional activation estimation requires a rest class")
    rest_index = labels.index("rest")
    rest_energies = [energies[index] for index in train_indices if targets[index] == rest_index]
    if not rest_energies:
        raise ArtifactError("training split has no rest windows for activation calibration")
    rest_floor = float(np.median(rest_energies))
    references: dict[str, float] = {}
    for class_index, label in enumerate(labels):
        if class_index == rest_index:
            continue
        class_energies = [
            energies[index] for index in train_indices if targets[index] == class_index
        ]
        if not class_energies:
            raise ArtifactError(f"training split has no {label} windows for activation calibration")
        reference = float(np.quantile(class_energies, reference_quantile))
        if reference <= rest_floor:
            raise ArtifactError(
                f"{label} energy reference {reference:.6g} does not exceed "
                f"the rest floor {rest_floor:.6g}"
            )
        references[label] = reference

    activations: list[float] = []
    for target, energy in zip(targets, energies, strict=True):
        label = labels[target]
        if label == "rest":
            activations.append(0.0)
            continue
        reference = references[label]
        activations.append(float(np.clip((energy - rest_floor) / (reference - rest_floor), 0, 1)))
    calibration = ActivationCalibration(
        method="causal_rms",
        window_samples=window_samples,
        rest_floor=rest_floor,
        reference_quantile=reference_quantile,
        class_references=tuple(references.items()),
    )
    return calibration, activations


class ActivationConditionedCrossEntropy(nn.Module):
    """Cross-entropy target construction that blends low-effort gestures toward rest."""
    """Interpolate only between rest and the prompted gesture near zero activation."""

    def __init__(self, rest_index: int, smoothing_threshold: float) -> None:
        """Record the rest class and activation boundary for target interpolation."""
        super().__init__()
        self.rest_index = rest_index
        self.smoothing_threshold = smoothing_threshold

    def targets(
        self, labels: torch.Tensor, activations: torch.Tensor, n_classes: int
    ) -> torch.Tensor:
        """Construct soft class targets from labels and proportional activation."""
        target = F.one_hot(labels, num_classes=n_classes).to(dtype=activations.dtype)
        active = labels != self.rest_index
        mix = torch.clamp(activations / self.smoothing_threshold, 0, 1)
        target[active, self.rest_index] = 1 - mix[active]
        target[active, labels[active]] = mix[active]
        return target

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        activations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute cross entropy against the activation-conditioned soft targets."""
        if activations is None:
            return F.cross_entropy(logits, labels)
        targets = self.targets(labels, activations, logits.shape[1])
        return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

    def accuracy_targets(
        self, labels: torch.Tensor, activations: torch.Tensor | None, n_classes: int
    ) -> torch.Tensor:
        """Return hard labels used only for interpretable accuracy reporting."""
        if activations is None:
            return labels
        targets = self.targets(labels, activations, n_classes)
        return targets.argmax(dim=1)


class EMGWindowDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Validated, grouped EMG windows loaded from QGrip Parquet projections."""
    """Read canonical QGrip Parquet sessions into grouped EMG windows."""

    def __init__(
        self,
        session_files: tuple[Path, ...],
        *,
        labels: tuple[str, ...],
        window_size: int,
        stride: int,
        channels: int,
        sample_rate_hz: float,
        include_activation: bool,
        activation_window_size: int,
    ) -> None:
        """Read compatible sessions and construct windows with provenance groups."""
        self.labels = labels
        self.label_to_index = {label: index for index, label in enumerate(labels)}
        self.include_activation = include_activation
        self.windows: list[np.ndarray] = []
        self.targets: list[int] = []
        self.activations: list[float] = []
        self.energies: list[float] = []
        self.groups: list[str] = []
        self.sample_rate_hz = sample_rate_hz
        self.channels = channels
        self.activation_window_size = activation_window_size
        for path in session_files:
            self._read_session(path, window_size, stride)
        if not self.windows:
            raise ArtifactError(
                f"no {window_size}-sample windows for {labels} in "
                f"{', '.join(str(path) for path in session_files)}"
            )
        if set(self.targets) != set(range(len(labels))):
            present = {labels[index] for index in self.targets}
            missing = sorted(set(labels).difference(present))
            raise ArtifactError(f"training data has no usable windows for classes: {missing}")

    def _read_session(self, path: Path, window_size: int, stride: int) -> None:
        """Validate one Parquet session and append its complete in-presentation windows."""
        if self.include_activation:
            metadata = pq.read_schema(path).metadata or {}
            method = metadata.get(b"qgrip.activation_energy.method")
            samples = metadata.get(b"qgrip.activation_energy.window_samples")
            if method != b"causal_rms" or samples is None:
                raise ArtifactError(f"{path}: missing causal RMS activation-energy metadata")
            try:
                stored_window_size = int(samples)
            except ValueError as exc:
                raise ArtifactError(f"{path}: invalid activation-energy window metadata") from exc
            if stored_window_size != self.activation_window_size:
                raise ArtifactError(
                    f"{path}: activation-energy window has {stored_window_size} samples; "
                    f"profile requires {self.activation_window_size}"
                )
        frame = pd.read_parquet(path)
        label_column = "gesture"
        channel_columns = [f"channel_{index}" for index in range(self.channels)]
        if not set(channel_columns).issubset(frame.columns):
            raise ArtifactError(f"{path}: missing {self.channels} QGrip channel columns")
        if label_column not in frame:
            raise ArtifactError(f"{path}: missing gesture column")
        if self.include_activation and "activation_energy" not in frame:
            raise ArtifactError(f"{path}: missing activation_energy column")
        if "sample_rate_hz" not in frame:
            raise ArtifactError(f"{path}: missing sample_rate_hz column")
        rates = set(frame["sample_rate_hz"].dropna().astype(float))
        if len(rates) != 1 or not math.isclose(rates.pop(), self.sample_rate_hz):
            raise ArtifactError(f"{path}: sample rate does not match the profile")
        frame = frame[frame[label_column].isin(self.labels)]
        required = {"capture_file", "trial", "sequence", "sample_index_in_packet"}
        if missing := required.difference(frame.columns):
            raise ArtifactError(f"{path}: missing QGrip columns {sorted(missing)}")
        group_columns = ["capture_file", "trial", label_column]
        sort_columns = ["sequence", "sample_index_in_packet"]
        for group_key, trial in frame.groupby(group_columns, sort=False, dropna=False):
            if sort_columns:
                trial = trial.sort_values(sort_columns)
            values = trial[channel_columns].to_numpy(dtype=np.float32)
            if len(values) < window_size:
                continue
            label = str(trial.iloc[0][label_column])
            group = f"{path.resolve()}::{group_key}"
            for start in range(0, len(values) - window_size + 1, stride):
                window = values[start : start + window_size]
                if not np.isfinite(window).all():
                    continue
                self.windows.append(window)
                self.targets.append(self.label_to_index[label])
                self.activations.append(1.0)
                if self.include_activation:
                    energy = float(trial.iloc[start + window_size - 1]["activation_energy"])
                    if not math.isfinite(energy):
                        self.windows.pop()
                        self.targets.pop()
                        self.activations.pop()
                        continue
                    self.energies.append(energy)
                self.groups.append(group)

    def __len__(self) -> int:
        """Return the number of constructed model windows."""
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        """Return one window, class target, activation, and provenance group."""
        values = torch.from_numpy(self.windows[index].copy())
        label = torch.tensor(self.targets[index], dtype=torch.long)
        if not self.include_activation:
            return values, label
        activation = torch.tensor(self.activations[index], dtype=torch.float32)
        return values, label, activation


def _train_epoch(
    model: BaseEMGClassifier,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    optimizer: torch.optim.Optimizer,
    class_loss: ActivationConditionedCrossEntropy,
    activation_loss: nn.Module | None,
    activation_weight: float,
    device: torch.device,
) -> tuple[float, float]:
    """Optimize one epoch and return mean loss plus hard-target accuracy."""
    model.train()
    total = 0.0
    correct = 0
    for batch in loader:
        values, labels = batch[:2]
        values, labels = values.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(values)
        logits = output[0] if isinstance(output, tuple) else output
        targets = batch[2].to(device) if activation_loss is not None else None
        loss = class_loss(logits, labels, targets)
        if activation_loss is not None:
            if not isinstance(output, tuple):
                raise RuntimeError("proportional model did not return activation")
            assert targets is not None
            loss = loss + activation_weight * activation_loss(output[1], targets)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * values.shape[0]
        accuracy_targets = class_loss.accuracy_targets(labels, targets, logits.shape[1])
        correct += int((logits.argmax(dim=1) == accuracy_targets).sum().item())
    if not isinstance(loader.dataset, Sized):
        raise TypeError("training dataset must have a finite size")
    count = len(loader.dataset)
    return total / count, correct / count


@torch.no_grad()
def _evaluate(
    model: BaseEMGClassifier,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    class_loss: ActivationConditionedCrossEntropy,
    activation_loss: nn.Module | None,
    activation_weight: float,
    device: torch.device,
) -> tuple[float, float, dict[str, float]]:
    """Evaluate one split without gradients and return loss, accuracy, and extras."""
    model.eval()
    total = 0.0
    correct = 0
    prompted_correct = 0
    predicted_activation: list[torch.Tensor] = []
    target_activation: list[torch.Tensor] = []
    for batch in loader:
        values, labels = batch[:2]
        values, labels = values.to(device), labels.to(device)
        output = model(values)
        logits = output[0] if isinstance(output, tuple) else output
        targets = batch[2].to(device) if activation_loss is not None else None
        loss = class_loss(logits, labels, targets)
        if activation_loss is not None:
            if not isinstance(output, tuple):
                raise RuntimeError("proportional model did not return activation")
            assert targets is not None
            loss = loss + activation_weight * activation_loss(output[1], targets)
            predicted_activation.append(output[1].cpu())
            target_activation.append(targets.cpu())
        total += float(loss.item()) * values.shape[0]
        predicted_classes = logits.argmax(dim=1)
        accuracy_targets = class_loss.accuracy_targets(labels, targets, logits.shape[1])
        correct += int((predicted_classes == accuracy_targets).sum().item())
        prompted_correct += int((predicted_classes == labels).sum().item())
    if not isinstance(loader.dataset, Sized):
        raise TypeError("validation dataset must have a finite size")
    count = len(loader.dataset)
    extra: dict[str, float] = {}
    extra["prompted_gesture_accuracy"] = prompted_correct / count
    if predicted_activation:
        predicted = torch.cat(predicted_activation).numpy()
        target = torch.cat(target_activation).numpy()
        error = predicted - target
        extra["activation_mae"] = float(np.abs(error).mean())
        extra["activation_rmse"] = float(np.sqrt(np.square(error).mean()))
    return total / count, correct / count, extra


def _summarize_split(
    labels: tuple[str, ...],
    targets: list[int],
    train_indices: list[int],
    validation_indices: list[int],
    window_size: int,
) -> TrainingSummary:
    """Count per-class windows on each side of the train/validation split."""
    training_counts = np.bincount(
        [targets[index] for index in train_indices], minlength=len(labels)
    )
    validation_counts = np.bincount(
        [targets[index] for index in validation_indices], minlength=len(labels)
    )
    classes = tuple(
        ClassSampleCount(label, int(training_counts[index]), int(validation_counts[index]))
        for index, label in enumerate(labels)
    )
    return TrainingSummary(
        training_samples=len(train_indices),
        validation_samples=len(validation_indices),
        window_size=window_size,
        classes=classes,
    )


class TorchTrainingService:
    """Optional-Torch implementation of QGrip's deterministic training workflow."""
    def __init__(self, options: TrainingConfig) -> None:
        """Retain immutable profile training options for one training invocation."""
        self.options = options

    def train(
        self,
        request: TrainingRequest,
        cancel: Any,
        metric: Callable[[EpochMetric], None] | None = None,
        summary: Callable[[TrainingSummary], None] | None = None,
    ) -> Path:
        """Train, validate, checkpoint, and optionally export a model artifact."""
        inputs = request.inputs or (latest_capture(request.profile, request.subject),)
        resolved: list[Path] = []
        for source in inputs:
            path = source.resolve()
            if path.name.endswith(".capture.jsonl.zst"):
                derived = parquet_path(path)
                path = (
                    derived
                    if derived.exists()
                    else export_capture(
                        path,
                        activation_energy_window_seconds=(
                            self.options.activation_energy_window_seconds
                        ),
                    )
                )
            if path.suffix != ".parquet" or not path.is_file():
                raise ArtifactError(f"training input is not Parquet: {path}")
            resolved.append(path)
        sample_rate = request.profile.device.sample_rate_hz
        window_size = max(8, round(sample_rate * self.options.training_window_seconds))
        dataset_stride = max(1, round(sample_rate * self.options.dataset_stride_seconds))
        activation_window_size = max(
            1, round(sample_rate * self.options.activation_energy_window_seconds)
        )
        n_fft_limit = 64 if sample_rate >= 400 else 32
        default_n_fft = 2 ** math.floor(math.log2(min(n_fft_limit, window_size)))
        n_fft = self.options.stft_n_fft or max(4, default_n_fft)
        if not 4 <= n_fft <= window_size:
            raise ArtifactError(
                "stft_n_fft must be at least 4 and no larger than the training window"
            )
        stft_hop_samples = self.options.stft_hop_samples or max(1, n_fft // 4)
        if not 1 <= stft_hop_samples <= n_fft:
            raise ArtifactError("stft_hop_samples must be between 1 and stft_n_fft")
        dataset = EMGWindowDataset(
            tuple(resolved),
            labels=request.profile.sgt.gestures,
            window_size=window_size,
            stride=dataset_stride,
            channels=request.profile.device.channels,
            sample_rate_hz=sample_rate,
            include_activation=request.proportional,
            activation_window_size=activation_window_size,
        )
        train_indices, validation_indices = self._split(dataset)
        activation_calibration: ActivationCalibration | None = None
        if request.proportional:
            activation_calibration, dataset.activations = _fit_activation_calibration(
                dataset.labels,
                dataset.targets,
                dataset.energies,
                train_indices,
                window_samples=activation_window_size,
                reference_quantile=self.options.activation_reference_quantile,
            )
        train_dataset = Subset(dataset, train_indices)
        validation_dataset = Subset(dataset, validation_indices)
        dataset_summary = _summarize_split(
            dataset.labels, dataset.targets, train_indices, validation_indices, window_size
        )
        LOGGER.info(
            "training on %d windows (%d train, %d validation) across %d classes",
            dataset_summary.total_samples,
            dataset_summary.training_samples,
            dataset_summary.validation_samples,
            len(dataset_summary.classes),
        )
        for entry in dataset_summary.classes:
            LOGGER.info(
                "  class %-16s %5d train / %5d validation",
                entry.label,
                entry.training,
                entry.validation,
            )
        if summary:
            summary(dataset_summary)
        generator = torch.Generator().manual_seed(self.options.seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.options.batch_size,
            shuffle=True,
            generator=generator,
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=self.options.batch_size, shuffle=False
        )
        torch.manual_seed(self.options.seed)
        np.random.seed(self.options.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_model(
            request.model,
            n_classes=len(dataset.labels),
            window_size=window_size,
            n_channels=dataset.channels,
            n_fft=n_fft,
            hop_length=stft_hop_samples,
            normalization=self.options.normalization.value,
            predict_activation=request.proportional,
            **dict(request.profile.model.architecture),
        )
        if model.normalization == "dataset_standardize":
            adapt_loader = DataLoader(train_dataset, batch_size=self.options.batch_size)
            model.preprocessor.adapt(batch[0] for batch in adapt_loader)
        model = model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.options.learning_rate,
            weight_decay=self.options.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.options.epochs)
        rest_index = dataset.label_to_index["rest"]
        class_loss = ActivationConditionedCrossEntropy(
            rest_index, self.options.activation_smoothing_threshold
        )
        activation_loss = nn.SmoothL1Loss() if request.proportional else None
        best_loss = float("inf")
        best_accuracy = 0.0
        best_state: dict[str, torch.Tensor] | None = None
        history: list[dict[str, object]] = []
        for epoch in range(1, self.options.epochs + 1):
            if cancel.is_set():
                raise InterruptedError("training cancelled")
            training_loss, training_accuracy = _train_epoch(
                model,
                train_loader,
                optimizer,
                class_loss,
                activation_loss,
                self.options.activation_loss_weight,
                device,
            )
            validation_loss, accuracy, extra = _evaluate(
                model,
                validation_loader,
                class_loss,
                activation_loss,
                self.options.activation_loss_weight,
                device,
            )
            scheduler.step()
            history.append(
                {
                    "epoch": epoch,
                    "training_loss": training_loss,
                    "training_accuracy": training_accuracy,
                    "validation_loss": validation_loss,
                    "accuracy": accuracy,
                    **extra,
                }
            )
            LOGGER.info(
                "epoch %d/%d train_loss=%.4f train_acc=%.3f val_loss=%.4f val_acc=%.3f",
                epoch,
                self.options.epochs,
                training_loss,
                training_accuracy,
                validation_loss,
                accuracy,
            )
            if metric:
                metric(
                    EpochMetric(
                        epoch,
                        validation_loss,
                        accuracy,
                        training_loss=training_loss,
                        training_accuracy=training_accuracy,
                    )
                )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_accuracy = accuracy
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
        if best_state is None:
            raise ArtifactError("training produced no checkpoint")
        model.load_state_dict(best_state)
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        output = subject_root(request.profile, request.subject) / "models" / run_id
        output.mkdir(parents=True, exist_ok=False)
        checkpoint_path = output / "model.pt"
        checkpoint: dict[str, Any] = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "model_state_dict": best_state,
            "model_name": request.model.value,
            "model_config": model.model_config,
            "labels": list(dataset.labels),
            "device": request.profile.device.kind.value,
            "sample_rate_hz": sample_rate,
            "activation_calibration": (
                {
                    "method": activation_calibration.method,
                    "window_seconds": self.options.activation_energy_window_seconds,
                    "window_samples": activation_calibration.window_samples,
                    "rest_floor": activation_calibration.rest_floor,
                    "reference_quantile": activation_calibration.reference_quantile,
                    "class_references": dict(activation_calibration.class_references),
                    "smoothing_threshold": self.options.activation_smoothing_threshold,
                }
                if activation_calibration is not None
                else None
            ),
            "val_loss": best_loss,
            "val_acc": best_accuracy,
            "inputs": [str(path) for path in resolved],
        }
        torch.save(checkpoint, checkpoint_path)
        metadata = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (output / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if self.options.export_onnx:
            try:
                export_model_to_onnx(model, output / "model.onnx")
            except Exception as exc:
                (output / "onnx-error.txt").write_text(str(exc), encoding="utf-8")
        return checkpoint_path

    def _split(self, dataset: EMGWindowDataset) -> tuple[list[int], list[int]]:
        """Create a seeded group split that keeps presentations wholly together."""
        indices = list(range(len(dataset)))
        groups = set(dataset.groups)
        if len(groups) >= 2:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=self.options.validation_fraction,
                random_state=self.options.seed,
            )
            training, validation = next(
                splitter.split(indices, dataset.targets, groups=dataset.groups)
            )
            return training.tolist(), validation.tolist()
        counts = np.bincount(dataset.targets, minlength=len(dataset.labels))
        stratify = dataset.targets if np.all(counts >= 2) else None
        training, validation = train_test_split(
            indices,
            test_size=max(1, round(len(indices) * self.options.validation_fraction)),
            random_state=self.options.seed,
            stratify=stratify,
        )
        return list(training), list(validation)
