"""Standalone, dashboard-independent Handi controller runtime."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Protocol

from qgrip.domain import (
    ControllerState,
    HandiConfig,
    Health,
    Prediction,
    QGripProfile,
)
from qgrip.errors import RpcError, ValidationError
from qgrip.rpc import MessagePackRpcClient
from qgrip.streaming import LiveEMGSession, PredictionDebouncer, sample_rates_match
from qgrip.workflows import InferenceService

LOGGER = logging.getLogger("qgrip.handi")
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
            positions=tuple((joint.name, joint.start) for joint in config.joints)
        )
        self._connected = False
        self._last_grip_switch: str | None = None
        self._switch_count = 0

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
        self._send_positions({joint.name: joint.start for joint in self.config.joints})

    def _send_positions(self, requested: dict[str, float]) -> None:
        """Clamp a full joint pose, transmit its ordered wire form, then publish it."""
        wire_positions = []
        clamped: dict[str, float] = {}
        current = dict(self.state.positions)
        for joint in self.config.joints:
            position = joint.clamp(requested.get(joint.name, current[joint.name]))
            clamped[joint.name] = position
            wire_positions.append(round(position))
        if not self.rpc.call("set_positions", wire_positions):
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

    def jog(self, name: str, delta: float) -> float:
        """Move one joint by a calibration-safe delta no larger than ``step``."""
        if abs(delta) > self.config.step:
            raise ValidationError(f"calibration jog is limited to {self.config.step}")
        return self.move(name, dict(self.state.positions)[name] + delta)

    def apply_grip(self, name: str) -> None:
        """Apply a named preset while retaining unspecified joints' current positions."""
        grip = next((item for item in self.config.grips if item.name == name), None)
        if grip is None:
            raise ValidationError(f"unknown grip: {name}")
        positions = dict(self.state.positions)
        positions.update(grip.positions)
        self._send_positions(positions)
        with self._lock:
            self._state = replace(self._state, grip=name)

    def apply_prediction(self, prediction: Prediction) -> None:
        """Map an accepted model output to a clamped incremental or preset motion."""
        action = dict(self.config.gesture_mapping).get(prediction.gesture)
        with self._lock:
            self._state = replace(self._state, prediction=prediction)
        if action is None:
            return
        if action in {"open", "close"}:
            direction = -1 if action == "open" else 1
            delta = direction * self.config.step * max(0, min(1, prediction.activation))
            for joint in self.config.joints:
                self.move(joint.name, dict(self.state.positions)[joint.name] + delta)
        elif any(grip.name == action for grip in self.config.grips):
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
