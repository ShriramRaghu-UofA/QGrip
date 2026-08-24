import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from qgrip.ml.models import (
    MODEL_NAMES,
    ONNXEMGClassifier,
    create_model,
    export_model_to_onnx,
    load_model_checkpoint,
)


class ModelTests(unittest.TestCase):
    def test_all_presets_predict_class_and_activation(self) -> None:
        values = torch.randn(2, 32, 8)
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model = create_model(
                    name,
                    n_classes=3,
                    window_size=32,
                    n_channels=8,
                    n_fft=16,
                    hop_length=4,
                    normalization="dataset_standardize",
                    predict_activation=True,
                )
                logits, activation = model(values)
                self.assertEqual(tuple(logits.shape), (2, 3))
                self.assertEqual(tuple(activation.shape), (2,))
                self.assertTrue(torch.all((activation >= 0) & (activation <= 1)))
                (logits.mean() + activation.mean()).backward()

    def test_cnn2d_uses_adaptive_max_pooling(self) -> None:
        model = create_model(
            "cnn2d",
            n_classes=3,
            window_size=32,
            n_channels=8,
            n_fft=16,
            hop_length=4,
            normalization="dataset_standardize",
            predict_activation=False,
        )

        self.assertIsInstance(model.classifier[0], torch.nn.AdaptiveMaxPool2d)

    def test_self_describing_checkpoint_round_trip(self) -> None:
        model = create_model(
            "dense",
            n_classes=3,
            window_size=16,
            n_channels=8,
            n_fft=8,
            hop_length=2,
            normalization="window_zscore",
            predict_activation=False,
        )
        checkpoint = {
            "checkpoint_version": 1,
            "model_state_dict": model.state_dict(),
            "model_name": "dense",
            "model_config": model.model_config,
            "labels": ["rest", "open", "close"],
            "sample_rate_hz": 200,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(checkpoint, path)
            loaded, metadata = load_model_checkpoint(path)
            self.assertEqual(metadata["labels"], checkpoint["labels"])
            self.assertEqual(tuple(loaded(torch.zeros(1, 16, 8)).shape), (1, 3))

    def test_checkpoint_version_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            for document, expected in (({}, "None"), ({"checkpoint_version": 2}, "2")):
                with self.subTest(checkpoint_version=expected):
                    torch.save(document, path)
                    with self.assertRaisesRegex(ValueError, f"checkpoint_version {expected}"):
                        load_model_checkpoint(path)

    def test_builtin_stft_exports_and_runs_in_onnx_runtime(self) -> None:
        model = create_model(
            "dense",
            n_classes=3,
            window_size=16,
            n_channels=8,
            n_fft=8,
            hop_length=2,
            normalization="window_zscore",
            predict_activation=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            export_model_to_onnx(model, path)
            logits, activation = ONNXEMGClassifier(path).predict(
                np.zeros((1, 16, 8), dtype=np.float32)
            )
            self.assertEqual(logits.shape, (1, 3))
            self.assertIsNotNone(activation)
            assert activation is not None
            self.assertEqual(activation.shape, (1,))


if __name__ == "__main__":
    unittest.main()
