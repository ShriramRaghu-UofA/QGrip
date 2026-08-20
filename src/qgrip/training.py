"""Self-contained EMG windowing and Torch classifier training."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Sized
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import nn
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
from qgrip.models import BaseEMGClassifier, create_model, export_model_to_onnx

LOGGER = logging.getLogger("qgrip.training")


class EMGWindowDataset(Dataset[tuple[torch.Tensor, ...]]):
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
    ) -> None:
        self.labels = labels
        self.label_to_index = {label: index for index, label in enumerate(labels)}
        self.include_activation = include_activation
        self.windows: list[np.ndarray] = []
        self.targets: list[int] = []
        self.activations: list[float] = []
        self.groups: list[str] = []
        self.sample_rate_hz = sample_rate_hz
        self.channels = channels
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
        frame = pd.read_parquet(path)
        label_column = "gesture"
        channel_columns = [f"channel_{index}" for index in range(self.channels)]
        if not set(channel_columns).issubset(frame.columns):
            raise ArtifactError(f"{path}: missing {self.channels} QGrip channel columns")
        if label_column not in frame:
            raise ArtifactError(f"{path}: missing gesture column")
        if "sample_rate" in frame:
            rates = set(frame["sample_rate"].dropna().astype(float))
            if rates and (len(rates) != 1 or not math.isclose(rates.pop(), self.sample_rate_hz)):
                raise ArtifactError(f"{path}: sample rate does not match the profile")
        if "presentation_stop_reason" in frame:
            frame = frame[frame["presentation_stop_reason"] == "completed"]
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
                activation = 1.0
                if self.include_activation:
                    activation_column = next(
                        (
                            name
                            for name in ("activation_measured", "activation", "activation_target")
                            if name in trial
                        ),
                        None,
                    )
                    if activation_column is None:
                        raise ArtifactError(f"{path}: proportional data has no activation column")
                    activation = float(trial.iloc[start + window_size - 1][activation_column])
                    if not math.isfinite(activation):
                        continue
                self.windows.append(window)
                self.targets.append(self.label_to_index[label])
                self.activations.append(float(np.clip(activation, 0, 1)))
                self.groups.append(group)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
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
    class_loss: nn.Module,
    activation_loss: nn.Module | None,
    activation_weight: float,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total = 0.0
    correct = 0
    for batch in loader:
        values, labels = batch[:2]
        values, labels = values.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(values)
        logits = output[0] if isinstance(output, tuple) else output
        loss = class_loss(logits, labels)
        if activation_loss is not None:
            if not isinstance(output, tuple):
                raise RuntimeError("proportional model did not return activation")
            loss = loss + activation_weight * activation_loss(output[1], batch[2].to(device))
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * values.shape[0]
        correct += int((logits.argmax(dim=1) == labels).sum().item())
    if not isinstance(loader.dataset, Sized):
        raise TypeError("training dataset must have a finite size")
    count = len(loader.dataset)
    return total / count, correct / count


@torch.no_grad()
def _evaluate(
    model: BaseEMGClassifier,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    class_loss: nn.Module,
    activation_loss: nn.Module | None,
    activation_weight: float,
    device: torch.device,
) -> tuple[float, float, dict[str, float]]:
    model.eval()
    total = 0.0
    correct = 0
    predicted_activation: list[torch.Tensor] = []
    target_activation: list[torch.Tensor] = []
    for batch in loader:
        values, labels = batch[:2]
        values, labels = values.to(device), labels.to(device)
        output = model(values)
        logits = output[0] if isinstance(output, tuple) else output
        loss = class_loss(logits, labels)
        if activation_loss is not None:
            if not isinstance(output, tuple):
                raise RuntimeError("proportional model did not return activation")
            targets = batch[2].to(device)
            loss = loss + activation_weight * activation_loss(output[1], targets)
            predicted_activation.append(output[1].cpu())
            target_activation.append(targets.cpu())
        total += float(loss.item()) * values.shape[0]
        correct += int((logits.argmax(dim=1) == labels).sum().item())
    if not isinstance(loader.dataset, Sized):
        raise TypeError("validation dataset must have a finite size")
    count = len(loader.dataset)
    extra: dict[str, float] = {}
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
    PRESET_VERSION = 1

    def __init__(self, options: TrainingConfig) -> None:
        self.options = options

    def train(
        self,
        request: TrainingRequest,
        cancel: Any,
        metric: Callable[[EpochMetric], None] | None = None,
        summary: Callable[[TrainingSummary], None] | None = None,
    ) -> Path:
        inputs = request.inputs or (latest_capture(request.profile, request.subject),)
        resolved: list[Path] = []
        for source in inputs:
            path = source.resolve()
            if path.name.endswith(".capture.jsonl.zst"):
                derived = parquet_path(path)
                path = derived if derived.exists() else export_capture(path)
            if path.suffix != ".parquet" or not path.is_file():
                raise ArtifactError(f"training input is not Parquet: {path}")
            resolved.append(path)
        sample_rate = request.profile.device.sample_rate_hz
        window_size = max(8, round(sample_rate * self.options.training_window_seconds))
        dataset_stride = max(1, round(sample_rate * self.options.dataset_stride_seconds))
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
        )
        train_indices, validation_indices = self._split(dataset)
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
        class_loss = nn.CrossEntropyLoss()
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
            "checkpoint_version": 2 if request.proportional else 1,
            "model_state_dict": best_state,
            "model_name": request.model.value,
            "model_config": model.model_config,
            "labels": list(dataset.labels),
            "label_to_idx": dataset.label_to_index,
            "window_size": window_size,
            "n_channels": dataset.channels,
            "channels": dataset.channels,
            "n_classes": len(dataset.labels),
            "n_fft": n_fft,
            "hop_length": stft_hop_samples,
            "device": request.profile.device.kind.value,
            "sample_rate": sample_rate,
            "sample_rate_hz": sample_rate,
            "normalization": model.normalization.value,
            "predict_activation": request.proportional,
            "preset_version": self.PRESET_VERSION,
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
