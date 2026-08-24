import unittest

import torch

from qgrip.core.domain import resolve_stft_dimensions
from qgrip.ml.training import (
    ActivationConditionedCrossEntropy,
    _fit_activation_calibration,
)


class ActivationCalibrationTests(unittest.TestCase):
    def test_default_stft_timing_aligns_at_supported_device_rates(self) -> None:
        expected = {
            200: (200, 20, 4),
            500: (500, 50, 10),
            1000: (1000, 100, 20),
            1600: (1600, 160, 32),
            2000: (2000, 200, 40),
        }
        for sample_rate_hz, dimensions in expected.items():
            with self.subTest(sample_rate_hz=sample_rate_hz):
                resolved = resolve_stft_dimensions(sample_rate_hz, 1.0, 0.1, 0.02)
                self.assertEqual(resolved, dimensions)
                window_size, n_fft, hop_length = resolved
                self.assertEqual((window_size - n_fft) % hop_length, 0)

    def test_stft_window_has_no_project_specific_duration_ceiling(self) -> None:
        self.assertEqual(resolve_stft_dimensions(200, 2.0, 0.5, 0.1), (400, 100, 20))

    def test_calibration_references_use_training_indices_only(self) -> None:
        calibration, activations = _fit_activation_calibration(
            ("rest", "open"),
            [0, 0, 1, 1, 0, 1],
            [1.0, 3.0, 6.0, 10.0, 100.0, 100.0],
            [0, 1, 2, 3],
            window_samples=20,
            reference_quantile=0.5,
        )

        self.assertEqual(calibration.rest_floor, 2.0)
        self.assertEqual(dict(calibration.class_references), {"open": 8.0})
        self.assertEqual(activations[4], 0.0)
        self.assertEqual(activations[5], 1.0)

    def test_soft_targets_move_mass_only_between_rest_and_gesture(self) -> None:
        loss = ActivationConditionedCrossEntropy(rest_index=0, smoothing_threshold=0.25)
        targets = loss.targets(
            torch.tensor([0, 1, 1, 2]),
            torch.tensor([0.0, 0.0, 0.125, 1.0]),
            n_classes=3,
        )

        torch.testing.assert_close(
            targets,
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
