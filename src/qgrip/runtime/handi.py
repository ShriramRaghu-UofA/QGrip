"""Standalone, dashboard-independent Handi controller runtime."""

from __future__ import annotations

import curses
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from qgrip.capture.rpc import MessagePackRpcClient
from qgrip.capture.streaming import LiveEMGSession, PredictionDebouncer, sample_rates_match
from qgrip.core.domain import (
    LED_MATRIX_PIXELS,
    ControllerState,
    GripPreset,
    HandiConfig,
    Health,
    JointLimit,
    Prediction,
    QGripProfile,
)
from qgrip.core.errors import RpcError, ValidationError
from qgrip.runtime.workflows import InferenceService

LOGGER = logging.getLogger("qgrip.runtime.handi")
HANDI_BRICK_REPOSITORY_URL = "https://github.com/YOUR-ORG/HANDI-BRICK-REPOSITORY"


class MotorRpc(Protocol):
    """Minimal Router transport contract needed by the safety controller."""

    def connect(self) -> None:
        """Open the Router transport."""
        ...

    def close(self) -> None:
        """Close the Router transport and unblock its callers."""
        ...

    def call(self, method: str, *args: object) -> object:
        """Synchronously invoke one Router method with serializable arguments."""
        ...


class HandController:
    """Owns hand state, clamps all commands, and maps predictions to safe movements."""

    def __init__(self, config: HandiConfig, rpc: MotorRpc) -> None:
        """Prepare safe initial state from verified limits; do not connect yet."""
        if not config.joints:
            raise ValidationError("Handi requires verified joint limits")
        self.config = config
        self.rpc = rpc
        self._lock = threading.RLock()
        self._state = ControllerState(
            positions=tuple((joint.name, joint.minimum) for joint in config.joints)
        )
        self._connected = False
        self._last_grip_step_gesture: str | None = None

    @property
    def state(self) -> ControllerState:
        """Return the latest immutable state under the controller lock."""
        with self._lock:
            return self._state

    def connect(self) -> None:
        """Open the validated Router transport before permitting movement."""
        self.rpc.connect()
        self._connected = True

    def apply_start_pose(self) -> None:
        """Send every configured joint's verified startup position."""
        if not self._connected:
            raise RpcError("startup validation has not completed")
        self._send_positions({joint.name: joint.minimum for joint in self.config.joints})

    def _send_positions(self, requested: dict[str, float]) -> None:
        """Clamp a full joint pose, transmit its ordered wire form, then publish it."""
        wire_positions = []
        clamped: dict[str, float] = {}
        current = dict(self.state.positions)
        for joint in self.config.joints:
            position = joint.clamp(requested.get(joint.name, current[joint.name]))
            clamped[joint.name] = position
            wire_positions.append(round(position))
        started = time.perf_counter()
        ok = self.rpc.call("set_positions", wire_positions)
        rpc_ms = (time.perf_counter() - started) * 1000
        LOGGER.info("set_positions rpc=%6.2fms", rpc_ms)
        if not ok:
            raise RpcError("set_positions was rejected by the MCU")
        with self._lock:
            self._state = replace(self._state, positions=tuple(clamped.items()))

    def move(self, name: str, requested: float) -> float:
        """Set one named joint safely and return the position actually accepted."""
        joint = next((item for item in self.config.joints if item.name == name), None)
        if joint is None:
            raise ValidationError(f"unknown joint: {name}")
        position = joint.clamp(requested)
        positions = dict(self.state.positions)
        positions[name] = position
        self._send_positions(positions)
        return position

    def move_many(self, requested: dict[str, float]) -> None:
        """Set several joints in one RPC call instead of one call per joint."""
        self._send_positions({**dict(self.state.positions), **requested})

    def jog(self, name: str, delta: float) -> float:
        """Move one joint by a calibration-safe delta no larger than ``step``."""
        if abs(delta) > self.config.step:
            raise ValidationError(f"calibration jog is limited to {self.config.step}")
        return self.move(name, dict(self.state.positions)[name] + delta)

    def cycle_grip(self, step: int) -> None:
        """Advance to the next/previous grip preset, wrapping around the list."""
        if not self.config.grips:
            raise ValidationError("Handi profile defines no grip presets to cycle")
        names = [grip.name for grip in self.config.grips]
        current = self.state.grip
        index = names.index(current) if current in names else -1
        self.apply_grip(names[(index + step) % len(names)], openness=self.state.openness)

    def apply_grip(self, name: str, *, openness: float = 1.0) -> None:
        """Snap to a named preset blended by ``openness`` (0.0 open .. 1.0 closed).

        Every joint the preset targets is blended between its own
        ``JointLimit.minimum`` (0.0, this joint's open/start position) and the
        preset's target (1.0) — the same "current grip, at the current
        openness" pose ``apply_prediction`` uses for open/close, so cycling
        grips doesn't force the hand fully closed. Joints the preset doesn't
        mention keep their current position.
        """
        grip = next((item for item in self.config.grips if item.name == name), None)
        if grip is None:
            raise ValidationError(f"unknown grip: {name}")
        openness = max(0.0, min(1.0, openness))
        joints_by_name = {joint.name: joint for joint in self.config.joints}
        positions = dict(self.state.positions)
        for joint_name, target in grip.positions:
            joint = joints_by_name[joint_name]
            positions[joint_name] = joint.minimum + openness * (target - joint.minimum)
        self._send_positions(positions)
        with self._lock:
            self._state = replace(self._state, grip=name, openness=openness)
        if grip.led_frame is not None:
            self._send_led_frame(grip.led_frame)

    def _send_led_frame(self, frame: tuple[int, ...]) -> None:
        """Best-effort push of a 104-pixel grayscale frame to the LED matrix.

        Mirrors the Router's ``set_led_frame`` RPC (grayscale 0-7 values, wire
        form is msgpack bin via ``bytes(frame)``). The matrix display is
        cosmetic, so a failure here is logged rather than raised — it must
        never abort a grip change that already moved the hand.
        """
        if len(frame) != LED_MATRIX_PIXELS:
            raise ValidationError(f"led_frame must have {LED_MATRIX_PIXELS} values")
        try:
            ok = self.rpc.call("set_led_frame", bytes(frame))
        except RpcError as exc:
            LOGGER.warning("set_led_frame failed: %s", exc)
            return
        if not ok:
            LOGGER.warning("set_led_frame rejected by MCU")

    def apply_prediction(self, prediction: Prediction) -> None:
        """Map an accepted model output to a clamped incremental or preset motion."""
        action = dict(self.config.gesture_mapping).get(prediction.gesture)
        with self._lock:
            self._state = replace(self._state, prediction=prediction)
        if action is None:
            self._last_grip_step_gesture = None
            return
        if action in {"open", "close"}:
            self._last_grip_step_gesture = None
            direction = -1 if action == "open" else 1
            activation = max(0.0, min(1.0, prediction.activation))
            openness = max(
                0.0,
                min(1.0, self.state.openness + direction * self.config.openness_step * activation),
            )
            grip = next(
                (item for item in self.config.grips if item.name == self.state.grip), None
            )
            if grip is None:
                # No grip is active yet - there's no preset target to blend toward, so
                # open/close has nothing to do but track openness for the next apply_grip.
                with self._lock:
                    self._state = replace(self._state, openness=openness)
                return
            joints_by_name = {joint.name: joint for joint in self.config.joints}
            positions = dict(self.state.positions)
            for joint_name, target in grip.positions:
                joint = joints_by_name[joint_name]
                positions[joint_name] = joint.minimum + openness * (target - joint.minimum)
            self._send_positions(positions)
            with self._lock:
                self._state = replace(self._state, openness=openness)
        elif action in {"next", "prev"}:
            if prediction.gesture == self._last_grip_step_gesture:
                return  # still the same contraction that already stepped - no-op
            self._last_grip_step_gesture = prediction.gesture
            self.cycle_grip(1 if action == "next" else -1)
        elif any(grip.name == action for grip in self.config.grips):
            self._last_grip_step_gesture = None
            self.apply_grip(action)

    def fail(self, message: str) -> None:
        """Publish an unhealthy stopped state after a runtime failure."""
        with self._lock:
            self._state = replace(self._state, running=False, healthy=False, error=message)

    def close(self) -> None:
        """Prevent further controller movement and close the underlying transport."""
        with self._lock:
            self._state = replace(self._state, running=False)
        self.rpc.close()
        self._connected = False


#: Jog step size, in raw servo units, per arrow-key press in the calibration
#: wizard. Shift+arrow (where the terminal reports it) jogs 10x.
JOG_STEP = 4
JOG_STEP_FAST = 40


def _get_position(rpc: MotorRpc, joint_id: int) -> tuple[int | None, str | None]:
    """Call get_position for a live read straight from the servo. Returns
    (position, error_message) — exactly one is None.
    """
    try:
        result = rpc.call("get_position", joint_id)
    except RpcError as exc:
        return None, f"get_position failed: {exc}"
    if not result or int(result[0]) < 0:
        return None, f"get_position({joint_id}) rejected by MCU"
    return int(result[0]), None


def _jog(rpc: MotorRpc, joint_id: int, delta: int, pos: int) -> tuple[int | None, str | None]:
    """Move joint_id to pos + delta via move_joint. Returns (new_goal, error_message)
    — exactly one is None.

    Returns the *commanded* goal (pos + delta), not a live get_position() readback -
    the servo travels at its configured profile velocity, so on a rapid key-repeat
    it's still physically in transit toward the previous goal when this jog fires;
    using its live present position as the next jog's base would make each step
    compound off a stale, lagging value and the commanded goal would barely creep
    forward (or stall outright) under fast repeat. Tracking the commanded goal
    instead means every keypress advances by exactly `delta` regardless of whether
    the servo has caught up yet.

    Curses has exclusive control of the terminal for the whole wizard, so errors are
    returned as strings to draw inside the UI rather than logged.
    """
    new_pos = pos + delta
    try:
        ok = bool(rpc.call("move_joint", joint_id, new_pos))
    except RpcError as exc:
        return None, f"move_joint failed: {exc}"
    if not ok:
        return None, f"move_joint({joint_id}, {new_pos}) rejected by MCU"
    return new_pos, None


def _retract_others(
    rpc: MotorRpc, joints: tuple[JointLimit, ...], except_joint_id: int
) -> str | None:
    """Move every other joint to its own known open/start endpoint (JointLimit.minimum),
    so the rest of the hand is out of the way while except_joint_id is being jogged.
    except_joint_id's own joint is always left alone.

    joint_id is 1-indexed position within `joints`, matching set_positions' wire
    order and the sketch's motor[] table (see HandController._send_positions).

    Returns None on success, else an error message.
    """
    for joint_id, limit in enumerate(joints, start=1):
        if joint_id == except_joint_id:
            continue
        try:
            ok = bool(rpc.call("move_joint", joint_id, limit.minimum))
        except RpcError as exc:
            return f"move_joint failed: {exc}"
        if not ok:
            return f"move_joint({joint_id}, {limit.minimum}) rejected by MCU"
    return None


def _calibrate_endpoint(
    stdscr: "curses._CursesWindow",
    rpc: MotorRpc,
    joint_id: int,
    joint_name: str,
    as_maximum: bool,
    start_pos: int,
) -> tuple[str, int]:
    """Jog one endpoint (EXTEND/minimum or FLEX/maximum) of one joint until 's' or 'd'.

    Returns (outcome, final_position); outcome is "saved", "discarded", or "quit"
    (Ctrl-C/q — aborts the whole wizard).
    """
    label = "FLEX (maximum)" if as_maximum else "EXTEND (minimum, also open/start position)"
    pos = start_pos
    status = ""
    while True:
        stdscr.erase()
        stdscr.addstr(0, 0, f"Joint {joint_id} ({joint_name})  —  finding {label}", curses.A_BOLD)
        stdscr.addstr(2, 0, f"  current position: {pos}")
        stdscr.addstr(3, 0, f"  started at:       {start_pos}")
        stdscr.addstr(5, 0, "  Left/Right  jog 1 step    Shift+Left/Right  jog 10 steps")
        stdscr.addstr(6, 0, "  s           record this position as the " + label)
        stdscr.addstr(7, 0, "  d           discard — jog back to start")
        stdscr.addstr(8, 0, "  q           quit wizard")
        if status:
            stdscr.addstr(10, 0, f"  ! {status}", curses.A_BOLD)
        stdscr.refresh()

        key = stdscr.getch()
        if key in (curses.KEY_LEFT, ord("h")):
            new_pos, status = _jog(rpc, joint_id, -JOG_STEP, pos)
            pos = new_pos if new_pos is not None else pos
            status = status or ""
        elif key in (curses.KEY_RIGHT, ord("l")):
            new_pos, status = _jog(rpc, joint_id, JOG_STEP, pos)
            pos = new_pos if new_pos is not None else pos
            status = status or ""
        elif key in (curses.KEY_SLEFT,):
            new_pos, status = _jog(rpc, joint_id, -JOG_STEP_FAST, pos)
            pos = new_pos if new_pos is not None else pos
            status = status or ""
        elif key in (curses.KEY_SRIGHT,):
            new_pos, status = _jog(rpc, joint_id, JOG_STEP_FAST, pos)
            pos = new_pos if new_pos is not None else pos
            status = status or ""
        elif key in (ord("s"), ord("S")):
            return "saved", pos
        elif key in (ord("d"), ord("D")):
            delta = start_pos - pos
            if delta:
                new_pos, status = _jog(rpc, joint_id, delta, pos)
                pos = new_pos if new_pos is not None else pos
            return "discarded", pos
        elif key in (ord("q"), ord("Q"), 3):  # 3 = Ctrl-C
            return "quit", pos


def _calibrate_motor(
    stdscr: "curses._CursesWindow",
    rpc: MotorRpc,
    joints: dict[str, JointLimit],
    joint_name: str,
) -> bool:
    """Walk one motor through EXTEND (minimum) then FLEX (maximum) endpoints.

    Retracts every other joint first (see _retract_others) so only the joint being
    calibrated is in the way. Saves into `joints[joint_name]` only if both endpoints
    end up actually saved this call (either one coming back "discarded" leaves the
    joint's existing limit untouched, since that endpoint is still whatever it was
    before). Returns False on quit.
    """
    order = list(joints)
    joint_id = order.index(joint_name) + 1

    retract_error = _retract_others(rpc, tuple(joints.values()), joint_id)
    if retract_error is not None:
        stdscr.erase()
        stdscr.addstr(0, 0, f"  ! {retract_error}", curses.A_BOLD)
        stdscr.addstr(2, 0, "  press any key to continue")
        stdscr.refresh()
        stdscr.getch()

    current = joints[joint_name]
    start_pos, _error = _get_position(rpc, joint_id)
    start_pos = start_pos if start_pos is not None else round(current.minimum)
    outcome, minimum_pos = _calibrate_endpoint(
        stdscr, rpc, joint_id, joint_name, False, start_pos
    )
    if outcome == "quit":
        return False
    if outcome != "saved":
        return True

    start_pos, _error = _get_position(rpc, joint_id)
    start_pos = start_pos if start_pos is not None else round(current.maximum)
    outcome, maximum_pos = _calibrate_endpoint(
        stdscr, rpc, joint_id, joint_name, True, start_pos
    )
    if outcome == "quit":
        return False
    if outcome != "saved":
        return True

    lo, hi = sorted((minimum_pos, maximum_pos))
    joints[joint_name] = JointLimit(joint_name, lo, hi)
    return True


def _list_menu(
    stdscr: "curses._CursesWindow",
    title: str,
    subtitle: str,
    items: list[tuple[int, str]],
) -> int | None:
    """Generic Up/Down/Enter list picker. items is [(id, label), ...].

    Returns the chosen id, or None if the user quit (q/Ctrl-C).
    """
    selected = 0
    while True:
        stdscr.erase()
        stdscr.addstr(0, 0, title, curses.A_BOLD)
        stdscr.addstr(1, 0, subtitle)
        for i, (item_id, label) in enumerate(items):
            attr = curses.A_REVERSE if i == selected else curses.A_NORMAL
            stdscr.addstr(3 + i, 2, f"{item_id}. {label}", attr)
        stdscr.refresh()

        key = stdscr.getch()
        if key in (curses.KEY_UP, ord("k")):
            selected = (selected - 1) % len(items)
        elif key in (curses.KEY_DOWN, ord("j")):
            selected = (selected + 1) % len(items)
        elif key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
            return items[selected][0]
        elif key in (ord("q"), ord("Q"), 3):
            return None


def _motor_menu(stdscr: "curses._CursesWindow", joints: dict[str, JointLimit]) -> str | None:
    """Arrow-key list of joints. Returns the chosen joint name, or None to quit."""
    names = list(joints)
    chosen_index = _list_menu(
        stdscr,
        "Handi calibration wizard — joint endpoints",
        "Up/Down to choose a joint, Enter to calibrate it, q to go back",
        list(enumerate(names)),
    )
    return None if chosen_index is None else names[chosen_index]


def _grip_menu(stdscr: "curses._CursesWindow", grips: dict[str, GripPreset]) -> str | None:
    """Arrow-key list of grip presets. Returns the chosen grip name, or None to quit."""
    names = list(grips)
    chosen_index = _list_menu(
        stdscr,
        "Handi calibration wizard — grip presets",
        "Up/Down to choose a preset, Enter to edit it, q to go back",
        list(enumerate(names)),
    )
    return None if chosen_index is None else names[chosen_index]


def _grip_digit_menu(
    stdscr: "curses._CursesWindow", grip_name: str, joint_names: list[str]
) -> str | None:
    """Arrow-key list of joints, scoped to one preset. Returns joint name, or None."""
    chosen_index = _list_menu(
        stdscr,
        f"Grip {grip_name} — edit a joint",
        "Up/Down to choose a joint, Enter to jog it, q for the preset list",
        list(enumerate(joint_names)),
    )
    return None if chosen_index is None else joint_names[chosen_index]


def _calibrate_grip_digit(
    stdscr: "curses._CursesWindow",
    controller: HandController,
    grip_name: str,
    joint_name: str,
    start_pos: float,
) -> str:
    """Jog one joint's position within one grip preset until 's' or 'd'.

    Returns "saved", "discarded", or "quit" (Ctrl-C/q — aborts the whole wizard).
    """
    pos = start_pos
    status = ""
    fast_step = min(JOG_STEP_FAST, controller.config.step)
    while True:
        stdscr.erase()
        stdscr.addstr(0, 0, f"Grip {grip_name}  —  {joint_name}", curses.A_BOLD)
        stdscr.addstr(2, 0, f"  current position: {pos:.0f}")
        stdscr.addstr(3, 0, f"  started at:       {start_pos:.0f}")
        stdscr.addstr(5, 0, "  Left/Right  jog 1 step    Shift+Left/Right  jog fast step")
        stdscr.addstr(6, 0, "  s           save this position for this joint in this preset")
        stdscr.addstr(7, 0, "  d           discard — jog back to start, keep old shape")
        stdscr.addstr(8, 0, "  q           quit wizard")
        if status:
            stdscr.addstr(10, 0, f"  ! {status}", curses.A_BOLD)
        stdscr.refresh()

        key = stdscr.getch()
        delta = 0
        if key in (curses.KEY_LEFT, ord("h")):
            delta = -JOG_STEP
        elif key in (curses.KEY_RIGHT, ord("l")):
            delta = JOG_STEP
        elif key in (curses.KEY_SLEFT,):
            delta = -fast_step
        elif key in (curses.KEY_SRIGHT,):
            delta = fast_step
        elif key in (ord("s"), ord("S")):
            return "saved"
        elif key in (ord("d"), ord("D")):
            if pos != start_pos:
                try:
                    pos = controller.move(joint_name, start_pos)
                except (RpcError, ValidationError) as exc:
                    status = str(exc)
            return "discarded"
        elif key in (ord("q"), ord("Q"), 3):  # 3 = Ctrl-C
            return "quit"

        if delta:
            try:
                pos = controller.jog(joint_name, delta)
                status = ""
            except (RpcError, ValidationError) as exc:
                status = str(exc)


def _calibrate_grip(
    stdscr: "curses._CursesWindow",
    controller: HandController,
    grips: dict[str, GripPreset],
    grip_name: str,
) -> bool:
    """Grip-editing loop: load the preset onto the hand, then repeatedly pick a
    joint to reshape. Edits `grips[grip_name]` in place. Returns False if the user
    quit the whole wizard (q/Ctrl-C at any point).
    """
    try:
        controller.apply_grip(grip_name)
    except (RpcError, ValidationError) as exc:
        stdscr.erase()
        stdscr.addstr(0, 0, f"  ! apply_grip failed: {exc}", curses.A_BOLD)
        stdscr.addstr(2, 0, "  press any key to continue")
        stdscr.refresh()
        stdscr.getch()

    joint_names = [joint.name for joint in controller.config.joints]
    while True:
        joint_name = _grip_digit_menu(stdscr, grip_name, joint_names)
        if joint_name is None:
            return True  # back to preset list, not a full quit
        start_pos = dict(controller.state.positions)[joint_name]
        outcome = _calibrate_grip_digit(stdscr, controller, grip_name, joint_name, start_pos)
        if outcome == "quit":
            return False
        if outcome == "saved":
            positions = dict(grips[grip_name].positions)
            positions[joint_name] = dict(controller.state.positions)[joint_name]
            grips[grip_name] = replace(grips[grip_name], positions=tuple(positions.items()))


#: Top-level wizard menu choices, see _wizard_main().
_MENU_JOINT_ENDPOINTS = 1
_MENU_GRIP_SHAPES = 2


def _wizard_main(
    stdscr: "curses._CursesWindow",
    rpc: MotorRpc,
    controller: HandController,
    joints: dict[str, JointLimit],
    grips: dict[str, GripPreset],
) -> None:
    curses.curs_set(0)
    stdscr.keypad(True)
    while True:
        choice = _list_menu(
            stdscr,
            "Handi calibration wizard",
            "Up/Down to choose, Enter to select, q to quit",
            [
                (_MENU_JOINT_ENDPOINTS, "Find joint range (extend/flex endpoints)"),
                (_MENU_GRIP_SHAPES, "Edit grip presets"),
            ],
        )
        if choice is None:
            return

        if choice == _MENU_JOINT_ENDPOINTS:
            while True:
                joint_name = _motor_menu(stdscr, joints)
                if joint_name is None:
                    break
                if not _calibrate_motor(stdscr, rpc, joints, joint_name):
                    return
        elif choice == _MENU_GRIP_SHAPES:
            while True:
                grip_name = _grip_menu(stdscr, grips)
                if grip_name is None:
                    break
                if not _calibrate_grip(stdscr, controller, grips, grip_name):
                    return


def run_interactive(profile: QGripProfile, rpc: MotorRpc, output: Path) -> None:
    """Curses calibration wizard with two modes, chosen from a top-level menu:

    1. Find joint range — pick a joint, jog it to each physical endpoint with the
       arrow keys (each jog is a move_joint call, position read back live via
       get_position), 's' to record that endpoint or 'd' to discard and jog back to
       where the endpoint started. Cycles EXTEND (becomes the joint's minimum —
       also its open/start position) then FLEX (maximum) for the chosen joint, then
       returns to the joint list. Nothing is written to the profile until the
       wizard exits — this only ever moves the servo and reads its live position,
       it's purely a way to find the numbers by eye/hand.

    2. Grip presets — pick a preset (its current shape is sent to the hand via the
       safety-clamped HandController), then pick a joint within it and jog that
       joint alone with the arrow keys, 's' to save its position into that preset,
       or 'd' to discard and jog back to where it started. Repeat for as many
       joints as you like, then 'q' for the preset list.

    'q' (or Ctrl-C) backs out one menu level; from the top-level menu it exits.

    On exit, the resulting joints/grips (unchanged unless you ran the matching
    mode) are written back into `profile.handi` and persisted to `output` via
    write_profile_atomic — see qgrip.core.profiles.
    """
    from qgrip.core.profiles import write_profile_atomic

    config = profile.handi
    assert config is not None
    joints = {joint.name: joint for joint in config.joints}
    grips = {grip.name: grip for grip in config.grips}
    controller = HandController(config, rpc)

    LOGGER.info("Starting calibration wizard...")
    try:
        controller.connect()
        controller.apply_start_pose()
        curses.wrapper(_wizard_main, rpc, controller, joints, grips)
    finally:
        new_handi = replace(config, joints=tuple(joints.values()), grips=tuple(grips.values()))
        new_profile = replace(profile, handi=new_handi)
        saved_path = write_profile_atomic(new_profile, output)
        LOGGER.info("Joint limits and grip presets saved to %s", saved_path)
        for joint in new_handi.joints:
            LOGGER.info("    %s: minimum=%g maximum=%g", joint.name, joint.minimum, joint.maximum)
        for grip in new_handi.grips:
            LOGGER.info("    %s: %s", grip.name, dict(grip.positions))
        controller.close()
        LOGGER.info("Calibration wizard exited")


class HandiRuntime:
    """Owns acquisition, inference, motor control, and exactly-once cleanup."""

    def __init__(
        self,
        profile: QGripProfile,
        model: str,
        *,
        rpc_factory: Callable[[str, float], MotorRpc] = MessagePackRpcClient,
    ) -> None:
        """Construct the independent runtime; validate configuration before ownership."""
        if profile.handi is None or not profile.handi.enabled:
            raise ValidationError("profile does not enable Handi")
        self.profile = profile
        self.model = InferenceService(model, profile.inference.backend)
        config = profile.handi
        self.controller = HandController(
            config, rpc_factory(config.rpc_socket, config.rpc_timeout_seconds)
        )
        self._stop = threading.Event()
        self._closed = False
        self._close_lock = threading.Lock()

    def validate(self) -> None:
        """Verify profile device identity against the loaded checkpoint metadata."""
        metadata = self.model.metadata
        if self.model.channels != self.profile.device.channels:
            raise ValidationError("model channel count does not match device")
        if not sample_rates_match(
            float(metadata["sample_rate_hz"]), self.profile.device.sample_rate_hz
        ):
            raise ValidationError("model sample rate does not match device")

    def start(self) -> None:
        """Validate, connect Router, and send a safe start pose atomically."""
        self.validate()
        try:
            self.controller.connect()
            self.controller.apply_start_pose()
            with self.controller._lock:
                self.controller._state = replace(self.controller._state, running=True)
        except BaseException:
            self.close()
            raise

    def run(self) -> None:
        """Own live acquisition/inference/control until stopped or an error occurs."""
        try:
            with LiveEMGSession(self.profile.device, self.profile.acquisition) as session:
                if session.channels != self.model.channels:
                    raise ValidationError("model channel count does not match live EMG stream")
                if not sample_rates_match(
                    session.sample_rate_hz, float(self.model.metadata["sample_rate_hz"])
                ):
                    raise ValidationError("model sample rate does not match live EMG stream")
                minimum_new_samples = max(
                    1,
                    round(session.sample_rate_hz * self.profile.inference.inference_period_seconds),
                )
                debouncer = PredictionDebouncer(self.profile.inference.switch_predictions)
                self.start()
                LOGGER.info("listening for gestures (Ctrl-C to stop)")
                next_inference_at = time.monotonic()
                while not self._stop.is_set():
                    wait_seconds = next_inference_at - time.monotonic()
                    if wait_seconds > 0:
                        self._stop.wait(
                            min(wait_seconds, self.profile.inference.maximum_wait_seconds)
                        )
                        continue
                    samples = session.next_window(self.model.window_size, minimum_new_samples)
                    if samples is not None:
                        prediction = self.model.predict(samples)
                        if prediction.confidence < self.profile.inference.confidence_gate:
                            prediction = replace(prediction, gesture="rest")
                        accepted = debouncer.accept(prediction)
                        LOGGER.info(
                            "gesture=%-16s confidence=%.2f activation=%.2f infer=%6.2fms%s",
                            prediction.gesture,
                            prediction.confidence,
                            prediction.activation,
                            prediction.latency_ms,
                            "  ACCEPTED" if accepted is not None else "",
                        )
                        if accepted is not None:
                            self.controller.apply_prediction(accepted)
                        next_inference_at += self.profile.inference.inference_period_seconds
                        if next_inference_at < time.monotonic():
                            next_inference_at = time.monotonic()
                    else:
                        self._stop.wait(self.profile.inference.idle_poll_seconds)
        except BaseException as exc:
            self.controller.fail(str(exc))
            raise
        finally:
            self.close()

    def stop(self) -> None:
        """Request cooperative exit from the live inference loop."""
        self._stop.set()

    def health(self) -> Health:
        """Summarize controller readiness for the optional observer API."""
        state = self.controller.state
        return Health(state.healthy, state.running, True, state.healthy, state.error or "")

    def close(self) -> None:
        """Perform exactly-once runtime cleanup without disabling servo torque."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        self.controller.close()


def main() -> int:
    """Expose the Handi-only console script without a separate entry module."""
    import sys

    from qgrip.runtime.cli import main as qgrip_main

    return qgrip_main(["handi", "run", *sys.argv[1:]])
