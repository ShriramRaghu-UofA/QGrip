import unittest

import torch

from qgrip.training import (
    ActivationConditionedCrossEntropy,
    _fit_activation_calibration,
)


class ActivationCalibrationTests(unittest.TestCase):
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
