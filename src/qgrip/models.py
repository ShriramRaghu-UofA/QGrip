"""Torch EMG classifiers migrated from the QGrip reference acquisition repository.

The architectures and checkpoint format originate in ``sifi-data-acquisition``.
They live here so QGrip training and inference have no runtime dependency on that
repository or its scripts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from qgrip.domain import ModelName, NormalizationMode

MODEL_NAMES: tuple[ModelName, ...] = tuple(ModelName)
CHECKPOINT_VERSION = 1


class EMGPreprocessor(nn.Module):
    """Normalize raw EMG and produce per-channel magnitude spectrograms."""

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        normalization: NormalizationMode | str,
        n_channels: int,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalization = NormalizationMode(normalization)
        self.n_channels = n_channels
        self.channel_mean: torch.Tensor = nn.Buffer(torch.zeros(n_channels))
        self.channel_scale: torch.Tensor = nn.Buffer(torch.ones(n_channels))
        self._window: torch.Tensor = nn.Buffer(torch.hann_window(n_fft))
        self._normalization_epsilon = 1e-5

    @torch.no_grad()
    def adapt(self, batches: Iterable[torch.Tensor]) -> None:
        """Fit fixed per-channel statistics from training data."""
        if self.normalization != NormalizationMode.DATASET_STANDARDIZE:
            raise RuntimeError("adapt() requires dataset_standardize normalization")
        total = torch.zeros(self.n_channels, dtype=torch.float64)
        total_squared = torch.zeros(self.n_channels, dtype=torch.float64)
        count = 0
        for batch in batches:
            values = torch.as_tensor(batch, dtype=torch.float64).reshape(-1, self.n_channels)
            total += values.sum(dim=0)
            total_squared += (values * values).sum(dim=0)
            count += values.shape[0]
        if count == 0:
            raise ValueError("cannot adapt normalization from an empty dataset")
        mean = total / count
        variance = (total_squared / count - mean * mean).clamp_min(0)
        scale = torch.sqrt(variance).clamp_min(torch.finfo(torch.float32).eps)
        self.channel_mean.copy_(mean.to(self.channel_mean))
        self.channel_scale.copy_(scale.to(self.channel_scale))

    def spectrogram(self, values: torch.Tensor) -> torch.Tensor:
        """Return ``(batch, channels, frequency, frames)`` magnitudes."""
        batch_size, samples, channels = values.shape
        if self.normalization == NormalizationMode.DATASET_STANDARDIZE:
            values = (values - self.channel_mean.view(1, 1, -1)) / self.channel_scale.view(1, 1, -1)
        signals = values.permute(0, 2, 1).reshape(batch_size * channels, samples)
        if self.normalization == NormalizationMode.WINDOW_ZSCORE:
            mean = signals.mean(dim=-1, keepdim=True)
            variance = signals.var(dim=-1, unbiased=False, keepdim=True)
            signals = (signals - mean) / torch.sqrt(variance + self._normalization_epsilon)
        elif self.normalization == NormalizationMode.SIGNED_8BIT:
            signals = signals / 128
        elif self.normalization != NormalizationMode.DATASET_STANDARDIZE:
            raise ValueError(f"unsupported EMG normalization {self.normalization!r}")
        spectrum = torch.stft(
            signals,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self._window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        magnitude = spectrum.abs()
        frequency_bins = self.n_fft // 2 + 1
        return magnitude.reshape(batch_size, channels, frequency_bins, magnitude.shape[-1])

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.spectrogram(values).permute(0, 3, 1, 2).flatten(start_dim=2)


class BaseEMGClassifier(nn.Module):
    def __init__(
        self,
        *,
        n_classes: int,
        window_size: int,
        n_channels: int,
        n_fft: int,
        hop_length: int,
        normalization: NormalizationMode | str,
        predict_activation: bool,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.n_channels = n_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalization = NormalizationMode(normalization)
        self.predict_activation = predict_activation
        self.preprocessor = EMGPreprocessor(n_fft, hop_length, self.normalization, n_channels)
        self._model_config: dict[str, Any] = {
            "n_classes": n_classes,
            "window_size": window_size,
            "n_channels": n_channels,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "normalization": self.normalization.value,
            "predict_activation": predict_activation,
        }

    @property
    def model_config(self) -> dict[str, Any]:
        return dict(self._model_config)

    @property
    def token_dim(self) -> int:
        return self.n_channels * (self.n_fft // 2 + 1)

    @property
    def n_frames(self) -> int:
        return 1 + (self.window_size - self.n_fft) // self.hop_length


class TransformerEMGClassifier(BaseEMGClassifier):
    def __init__(
        self,
        *,
        d_model: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0,
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self._model_config.update(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.input_proj = nn.Linear(self.token_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_frames, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, config["n_classes"])
        )
        self.activation_head = nn.Linear(d_model, 1) if self.predict_activation else None

    def forward(self, values: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        tokens = self.input_proj(self.preprocessor(values)) + self.pos_embed
        features = self.transformer(tokens).mean(dim=1)
        logits = self.classifier(features)
        if self.activation_head is None:
            return logits
        return logits, torch.sigmoid(self.activation_head(features)).squeeze(1)


class CNN1DEMGClassifier(BaseEMGClassifier):
    def __init__(self, *, hidden_channels: int = 64, dropout: float = 0, **config: Any) -> None:
        super().__init__(**config)
        self._model_config.update(hidden_channels=hidden_channels, dropout=dropout)
        self.features = nn.Sequential(
            nn.Conv1d(self.token_dim, hidden_channels, 5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, config["n_classes"]),
        )
        self.activation_head = nn.Linear(hidden_channels, 1) if self.predict_activation else None

    def forward(self, values: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        encoded = self.features(self.preprocessor(values).transpose(1, 2))
        features = self.classifier[:-1](encoded)
        logits = self.classifier[-1](features)
        if self.activation_head is None:
            return logits
        return logits, torch.sigmoid(self.activation_head(features)).squeeze(1)


class CNN2DEMGClassifier(BaseEMGClassifier):
    def __init__(
        self,
        *,
        hidden_channels: int = 16,
        dropout: float = 0,
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self._model_config.update(hidden_channels=hidden_channels, dropout=dropout)
        self.features = nn.Sequential(
            nn.Conv2d(self.n_channels, hidden_channels, 3, padding=1, stride=2),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 3, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, config["n_classes"]),
        )
        self.activation_head = (
            nn.Linear(hidden_channels * 2, 1) if self.predict_activation else None
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        encoded = self.features(self.preprocessor.spectrogram(values))
        features = self.classifier[:-1](encoded)
        logits = self.classifier[-1](features)
        if self.activation_head is None:
            return logits
        return logits, torch.sigmoid(self.activation_head(features)).squeeze(1)


class DenseEMGClassifier(BaseEMGClassifier):
    def __init__(self, *, hidden_dim: int = 256, dropout: float = 0, **config: Any) -> None:
        super().__init__(**config)
        self._model_config.update(hidden_dim=hidden_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_frames * self.token_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, config["n_classes"]),
        )
        self.activation_head = nn.Linear(hidden_dim, 1) if self.predict_activation else None

    def forward(self, values: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        encoded = self.preprocessor(values)
        features = self.classifier[:-1](encoded)
        logits = self.classifier[-1](features)
        if self.activation_head is None:
            return logits
        return logits, torch.sigmoid(self.activation_head(features)).squeeze(1)


MODEL_CLASSES: dict[ModelName, type[BaseEMGClassifier]] = {
    ModelName.TRANSFORMER: TransformerEMGClassifier,
    ModelName.CNN1D: CNN1DEMGClassifier,
    ModelName.CNN2D: CNN2DEMGClassifier,
    ModelName.DENSE: DenseEMGClassifier,
}


def create_model(model_name: ModelName | str, **config: Any) -> BaseEMGClassifier:
    try:
        resolved_name = ModelName(model_name)
    except ValueError as exc:
        raise ValueError(f"unknown model {model_name!r}; choose from {MODEL_NAMES}") from exc
    return MODEL_CLASSES[resolved_name](**config)


def _validate_checkpoint_version(checkpoint: Mapping[str, Any]) -> None:
    checkpoint_version = checkpoint.get("checkpoint_version")
    if checkpoint_version != CHECKPOINT_VERSION:
        raise ValueError(
            f"checkpoint has unsupported checkpoint_version {checkpoint_version!r}; "
            f"expected {CHECKPOINT_VERSION}"
        )


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    checkpoint = cast(dict[str, Any], torch.load(path, map_location=device, weights_only=True))
    _validate_checkpoint_version(checkpoint)
    return checkpoint


def checkpoint_model_config(checkpoint: Mapping[str, Any]) -> tuple[ModelName, dict[str, Any]]:
    _validate_checkpoint_version(checkpoint)
    raw_model_name = checkpoint.get("model_name")
    try:
        model_name = ModelName(raw_model_name)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint has unsupported model_name {raw_model_name!r}") from exc
    config = checkpoint.get("model_config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint has no model_config")
    return model_name, dict(cast(Mapping[str, Any], config))


def load_model_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[BaseEMGClassifier, dict[str, Any]]:
    checkpoint = load_checkpoint(path, device)
    model_name, config = checkpoint_model_config(checkpoint)
    model = create_model(model_name, **config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, checkpoint


def export_model_to_onnx(model: BaseEMGClassifier, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    example = torch.zeros(1, model.window_size, model.n_channels, device=device)
    output_names = ["class_logits", "activation"] if model.predict_activation else ["logits"]
    model.eval()
    torch.onnx.export(
        model,
        (example,),
        output,
        input_names=["raw_emg"],
        output_names=output_names,
        opset_version=18,
        dynamo=True,
        external_data=False,
    )


class ONNXEMGClassifier:
    """ONNX Runtime adapter with the same raw-window contract as Torch."""

    def __init__(self, path: str | Path, *, prefer_cuda: bool = False) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX inference requires the qgrip onnx extra") from exc
        providers = ["CPUExecutionProvider"]
        if prefer_cuda and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self._session = ort.InferenceSession(str(path), providers=providers)
        input_metadata = self._session.get_inputs()[0]
        outputs = self._session.get_outputs()
        self._input_name = input_metadata.name
        self._output_names = [item.name for item in outputs]
        self.input_shape = tuple(input_metadata.shape)
        self.providers = tuple(self._session.get_providers())

    def predict(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        inputs = np.ascontiguousarray(windows, dtype=np.float32)
        outputs = self._session.run(self._output_names, {self._input_name: inputs})
        logits = cast(np.ndarray, outputs[0])
        activation = cast(np.ndarray, outputs[1]) if len(outputs) > 1 else None
        return logits, activation
