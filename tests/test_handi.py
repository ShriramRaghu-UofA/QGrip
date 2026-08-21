import unittest

from qgrip.core.domain import GripPreset, HandiConfig, JointLimit, Prediction
from qgrip.core.errors import ValidationError
from qgrip.runtime.handi import HandController


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
            joints=(
                JointLimit("thumb", 10, 20, 15, open_position=10),
                JointLimit("index", 100, 200, 150, open_position=100),
            ),
        )
        self.controller = HandController(self.config, self.rpc)
        self.controller.connect()
        self.controller.apply_start_pose()

    def test_all_positions_are_batched_and_clamped(self) -> None:
        self.assertEqual(self.controller.move("thumb", 999), 20)
        self.assertEqual(self.rpc.calls[-1], ("set_positions", ([20, 150],)))

    def test_open_close_is_noop_without_an_active_grip(self) -> None:
        # No grip has been applied yet, so there's no preset target to blend
        # toward - open/close has nothing to move, only openness to track.
        self.rpc.calls.clear()
        self.controller.apply_prediction(Prediction("close", 1, 1))
        self.assertEqual(self.rpc.calls, [])
        self.assertEqual(self.controller.state.openness, 1.0)

    def test_jog_is_bounded(self) -> None:
        with self.assertRaises(ValidationError):
            self.controller.jog("thumb", 11)


class HandControllerLedFrameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rpc = FakeRpc()
        self.led_frame = tuple(range(8)) * 13  # 104 values, 0-7 repeating
        self.config = HandiConfig(
            enabled=True,
            step=10,
            joints=(JointLimit("thumb", 10, 20, 15, open_position=10),),
            grips=(GripPreset("fist", (("thumb", 20),), self.led_frame),),
        )
        self.controller = HandController(self.config, self.rpc)
        self.controller.connect()
        self.controller.apply_start_pose()

    def test_apply_grip_sends_led_frame_after_positions(self) -> None:
        self.controller.apply_grip("fist")
        methods = [call[0] for call in self.rpc.calls]
        self.assertEqual(methods[-2:], ["set_positions", "set_led_frame"])
        self.assertEqual(self.rpc.calls[-1], ("set_led_frame", (bytes(self.led_frame),)))

    def test_grip_without_led_frame_does_not_call_set_led_frame(self) -> None:
        config = HandiConfig(
            enabled=True,
            step=10,
            joints=(JointLimit("thumb", 10, 20, 15, open_position=10),),
            grips=(GripPreset("open", (("thumb", 10),)),),
        )
        controller = HandController(config, self.rpc)
        controller.connect()
        controller.apply_start_pose()
        self.rpc.calls.clear()
        controller.apply_grip("open")
        self.assertEqual([call[0] for call in self.rpc.calls], ["set_positions"])

    def test_set_led_frame_failure_does_not_raise(self) -> None:
        class RejectingRpc(FakeRpc):
            def call(self, method: str, *args: object) -> object:
                self.calls.append((method, args))
                return method != "set_led_frame"

        controller = HandController(self.config, RejectingRpc())
        controller.connect()
        controller.apply_start_pose()
        controller.apply_grip("fist")  # must not raise


class HandControllerGripRelativeOpennessTests(unittest.TestCase):
    """Open/close must blend toward the *active grip's own* preset shape,
    not a generic full-hand close, so different grips stay visually distinct
    at partial openness instead of collapsing through one shared pose."""

    def setUp(self) -> None:
        self.rpc = FakeRpc()
        self.config = HandiConfig(
            enabled=True,
            step=10,
            openness_step=0.5,
            joints=(
                JointLimit("thumb", 0, 100, 0, open_position=0),
                JointLimit("index", 0, 100, 0, open_position=0),
            ),
            grips=(
                GripPreset("pinch", (("thumb", 40), ("index", 80))),
                GripPreset("fist", (("thumb", 100), ("index", 100))),
            ),
        )
        self.controller = HandController(self.config, self.rpc)
        self.controller.connect()
        self.controller.apply_start_pose()

    def test_apply_grip_blends_from_current_openness(self) -> None:
        self.controller.apply_grip("pinch")  # openness defaults to 1.0 -> full preset
        self.assertEqual(dict(self.controller.state.positions)["thumb"], 40)
        self.assertEqual(dict(self.controller.state.positions)["index"], 80)

    def test_close_blends_toward_active_grips_own_shape(self) -> None:
        self.controller.apply_grip("pinch", openness=0.0)
        self.controller.apply_prediction(Prediction("close", 1, 1))  # +0.5 openness
        positions = dict(self.controller.state.positions)
        self.assertEqual(positions["thumb"], 20)  # halfway 0 -> 40
        self.assertEqual(positions["index"], 40)  # halfway 0 -> 80

    def test_different_grips_stay_distinct_at_partial_openness(self) -> None:
        self.controller.apply_grip("pinch", openness=0.5)
        pinch_positions = dict(self.controller.state.positions)
        self.controller.apply_grip("fist", openness=0.5)
        fist_positions = dict(self.controller.state.positions)
        self.assertNotEqual(pinch_positions["index"], fist_positions["index"])

    def test_open_close_does_not_exceed_bounds(self) -> None:
        self.controller.apply_grip("fist", openness=1.0)
        self.controller.apply_prediction(Prediction("close", 1, 1))
        self.assertEqual(self.controller.state.openness, 1.0)
        for _ in range(4):
            self.controller.apply_prediction(Prediction("open", 1, 1))
        self.assertEqual(self.controller.state.openness, 0.0)
