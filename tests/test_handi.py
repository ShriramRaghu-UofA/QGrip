import unittest

from qgrip.domain import HandiConfig, JointLimit, Prediction
from qgrip.errors import ValidationError
from qgrip.handi import HandController


class FakeRpc:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.closed = 0

    def connect(self) -> None:
        pass

    def close(self) -> None:
        self.closed += 1

    def call(self, method: str, *args: object) -> object:
        self.calls.append((method, args))
        return True


class HandControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rpc = FakeRpc()
        self.config = HandiConfig(
            enabled=True,
            step=10,
            joints=(JointLimit("thumb", 10, 20, 15), JointLimit("index", 100, 200, 150)),
        )
        self.controller = HandController(self.config, self.rpc)
        self.controller.connect()
        self.controller.apply_start_pose()

    def test_all_positions_are_batched_and_clamped(self) -> None:
        self.assertEqual(self.controller.move("thumb", 999), 20)
        self.assertEqual(self.rpc.calls[-1], ("set_positions", ([20, 150],)))

    def test_activation_scales_open_close_step(self) -> None:
        self.controller.apply_prediction(Prediction("close", 1, 0.5))
        self.assertEqual(dict(self.controller.state.positions)["thumb"], 20)
        self.controller.apply_prediction(Prediction("open", 1, 1))
        self.assertEqual(dict(self.controller.state.positions)["thumb"], 10)

    def test_jog_is_bounded(self) -> None:
        with self.assertRaises(ValidationError):
            self.controller.jog("thumb", 11)
