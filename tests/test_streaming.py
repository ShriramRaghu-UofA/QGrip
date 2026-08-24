import unittest

import numpy as np
from sifi_streamer.acquisition.reader import SignalWindow

from qgrip.capture.streaming import (
    LiveEMGSession,
    MyoAcquisitionDevice,
    MyoPacket,
    PredictionDebouncer,
)
from qgrip.core.domain import DeviceConfig, DeviceKind, Prediction


class _Reader:
    def __init__(self) -> None:
        self._windows = [
            SignalWindow(0, 2, np.array([0.0, 0.1]), np.array([[1.0], [2.0]]), np.ones(2)),
            SignalWindow(2, 4, np.array([0.2, 0.3]), np.array([[3.0], [4.0]]), np.ones(2)),
            SignalWindow(4, 5, np.array([0.4]), np.array([[5.0]]), np.ones(1)),
            SignalWindow(5, 6, np.array([0.5]), np.array([[6.0]]), np.ones(1)),
        ]

    def read_since(self, cursor: int) -> SignalWindow | None:
        return self._windows.pop(0) if self._windows else None


class LiveWindowTests(unittest.TestCase):
    def test_window_emits_only_after_initial_fill_and_new_sample_threshold(self) -> None:
        session = LiveEMGSession(DeviceConfig(kind=DeviceKind.SYNTHETIC, channels=1))
        session._reader = _Reader()
        self.assertIsNone(session.next_window(4, 2))
        np.testing.assert_array_equal(
            session.next_window(4, 2), np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        )
        self.assertIsNone(session.next_window(4, 2))
        np.testing.assert_array_equal(
            session.next_window(4, 2), np.array([[3.0], [4.0], [5.0], [6.0]], dtype=np.float32)
        )

    def test_myo_adapter_uses_the_profile_sample_rate(self) -> None:
        config = DeviceConfig(kind=DeviceKind.MYO_BLE, sample_rate_hz=250)
        device = MyoAcquisitionDevice(config)
        self.assertEqual(device.streams[0].nominal_rate_hz, 250)
        self.assertEqual(device.device_info["sample_rate_hz"], 250)
        self.assertEqual(MyoPacket(0, (0.0,) * 8, 250).reported_rate_hz, 250)


class PredictionDebouncerTests(unittest.TestCase):
    def test_requires_consecutive_predictions_before_switching_gestures(self) -> None:
        debouncer = PredictionDebouncer(2)
        open_prediction = Prediction("open", 0.9, 0.5)
        close_prediction = Prediction("close", 0.9, 0.5)
        self.assertIsNone(debouncer.accept(open_prediction))
        self.assertEqual(debouncer.accept(open_prediction), open_prediction)
        self.assertIsNone(debouncer.accept(close_prediction))
        self.assertEqual(debouncer.accept(open_prediction), open_prediction)
        self.assertIsNone(debouncer.accept(close_prediction))
        self.assertEqual(debouncer.accept(close_prediction), close_prediction)
