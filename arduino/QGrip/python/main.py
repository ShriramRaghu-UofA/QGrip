import time

from arduino.app_utils import App, Bridge, Logger

logger = Logger("myohandihand")

# Set to True to cycle through a couple of test hand shapes on a timer instead of
# waiting for RPC calls - lets the sketch/wiring be bench-tested without anything
# driving the Bridge. Set back to False before shipping/normal use.
TEST_MODE = False

DIGIT_NUM = 6
LED_MATRIX_PIXELS = 104

# [thumb_rotate, thumb_flex, index, middle, ring, baby]
OPEN_HAND = (0, 0, 0, 0, 0, 0)
CLOSED_HAND = (2700,) * DIGIT_NUM  # bench-test-only shape; not a real drive limit

TEST_SHAPES = (OPEN_HAND, CLOSED_HAND)
TEST_INTERVAL_S = 3.0


def set_positions(positions) -> bool:
    """Drive every joint to `positions` (6 raw DYNAMIXEL goals, one per joint - see
    the sketch's Definitions comment for joint order). No range checking - sent
    as-is; drive-limit gating is the caller's responsibility (see HandController
    in src/qgrip/runtime/handi.py). Returns True if the sketch accepted the command.
    """
    ok = Bridge.call("set_positions", list(positions))
    if not ok:
        logger.warning(f"set_positions({list(positions)}) rejected by MCU")
    return ok


def set_led_frame(led_frame) -> bool:
    """Draw `led_frame` (104 grayscale 0-7 values, row-major) on the LED matrix,
    sent as bytes (msgpack bin). Returns True if the sketch accepted the command.
    """
    ok = Bridge.call("set_led_frame", bytes(led_frame))
    if not ok:
        logger.warning("set_led_frame(...) rejected by MCU")
    return ok


def move_joint(joint_id: int, position: int) -> bool:
    """Move a single digit (joint id 1-6) to a goal position (raw DYNAMIXEL units,
    sent as-is - no range checking).

    Bench-testing/manual-move helper, independent of set_positions(). Returns True if
    the sketch accepted the command (valid, enabled joint id).
    """
    ok = Bridge.call("move_joint", joint_id, position)
    if not ok:
        logger.warning(f"move_joint({joint_id}, {position}) rejected by MCU")
    return ok


def get_position(joint_id: int) -> tuple[int, int]:
    """Read digit `joint_id`'s live present position and velocity straight from the
    servo (not the last commanded value). Returns (-1, -1) if joint_id was
    invalid/disabled.
    """
    position, velocity = Bridge.call("get_position", joint_id)
    return position, velocity


def test_shape_cycle_loop():
    """Cycle through every shape in TEST_SHAPES, logging each result.

    Runs in place of the normal (idle) user_loop when TEST_MODE is True, so you can
    watch it end-to-end with `arduino-app-cli app logs --follow` on the bench
    without needing an external RPC caller.
    """
    for positions in TEST_SHAPES:
        ok = set_positions(positions)
        logger.info(f"[test] positions -> {positions}: {'ok' if ok else 'FAILED'}")
        time.sleep(TEST_INTERVAL_S)


if TEST_MODE:
    logger.info("TEST_MODE=True: cycling test shapes instead of waiting for RPC")
    App.run(user_loop=test_shape_cycle_loop)
else:
    App.run()
