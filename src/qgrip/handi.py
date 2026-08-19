"""Standalone, dashboard-independent Handi controller runtime."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import replace
from typing import Protocol

from qgrip.devices import SignalDevice, create_device
from qgrip.domain import (
    ControllerState,
    DeviceConfig,
    HandiConfig,
    Health,
    Prediction,
    QGripProfile,
)
from qgrip.errors import RpcError, ValidationError
from qgrip.rpc import MessagePackRpcClient
from qgrip.workflows import InferenceService

LOGGER = logging.getLogger("qgrip.handi")
HANDI_BRICK_REPOSITORY_URL = "https://github.com/YOUR-ORG/HANDI-BRICK-REPOSITORY"


class MotorRpc(Protocol):
    def connect(self) -> None: ...

    def close(self) -> None: ...

    def call(self, method: str, *args: object) -> object: ...


class HandController:
    """Owns hand state, clamps all commands, and maps predictions to safe movements."""

    def __init__(self, config: HandiConfig, rpc: MotorRpc) -> None:
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
        with self._lock:
            return self._state

    def connect(self) -> None:
        self.rpc.connect()
        self._connected = True

    def apply_start_pose(self) -> None:
        if not self._connected:
            raise RpcError("startup validation has not completed")
        self._send_positions({joint.name: joint.start for joint in self.config.joints})

    def _send_positions(self, requested: dict[str, float]) -> None:
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
        joint = next((item for item in self.config.joints if item.name == name), None)
        if joint is None:
            raise ValidationError(f"unknown joint: {name}")
        position = joint.clamp(requested)
        positions = dict(self.state.positions)
        positions[name] = position
        self._send_positions(positions)
        return position

    def jog(self, name: str, delta: float) -> float:
        if abs(delta) > self.config.step:
            raise ValidationError(f"calibration jog is limited to {self.config.step}")
        return self.move(name, dict(self.state.positions).get(name, 0) + delta)

    def apply_grip(self, name: str) -> None:
        grip = next((item for item in self.config.grips if item.name == name), None)
        if grip is None:
            raise ValidationError(f"unknown grip: {name}")
        positions = dict(self.state.positions)
        positions.update(grip.positions)
        self._send_positions(positions)
        with self._lock:
            self._state = replace(self._state, grip=name)

    def apply_prediction(self, prediction: Prediction) -> None:
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
        with self._lock:
            self._state = replace(self._state, running=False, healthy=False, error=message)

    def close(self) -> None:
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
        device_factory: Callable[[DeviceConfig], SignalDevice] = create_device,
        rpc_factory: Callable[[str, float], MotorRpc] = MessagePackRpcClient,
    ) -> None:
        if profile.handi is None or not profile.handi.enabled:
            raise ValidationError("profile does not enable Handi")
        self.profile = profile
        self.model = InferenceService(model)
        self.device = device_factory(profile.device)
        config = profile.handi
        self.controller = HandController(
            config, rpc_factory(config.rpc_socket, config.rpc_timeout_seconds)
        )
        self._stop = threading.Event()
        self._closed = False
        self._close_lock = threading.Lock()

    def validate(self) -> None:
        metadata = self.model.metadata
        if int(metadata["channels"]) != self.profile.device.channels:
            raise ValidationError("model channel count does not match device")
        if float(metadata["sample_rate_hz"]) != self.profile.device.sample_rate_hz:
            raise ValidationError("model sample rate does not match device")

    def start(self) -> None:
        self.validate()
        try:
            self.device.connect()
            self.controller.connect()
            self.controller.apply_start_pose()
            with self.controller._lock:
                self.controller._state = replace(self.controller._state, running=True)
        except BaseException:
            self.close()
            raise

    def run(self) -> None:
        self.start()
        try:
            while not self._stop.is_set():
                packet = self.device.read(
                    max(
                        1, int(self.device.sample_rate_hz * self.profile.inference.interval_seconds)
                    )
                )
                prediction = self.model.predict(packet.samples)
                if prediction.confidence >= self.profile.inference.confidence_gate:
                    self.controller.apply_prediction(prediction)
        except BaseException as exc:
            self.controller.fail(str(exc))
            raise
        finally:
            self.close()

    def stop(self) -> None:
        self._stop.set()

    def health(self) -> Health:
        state = self.controller.state
        return Health(state.healthy, state.running, True, state.healthy, state.error or "")

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        try:
            self.device.close()
        finally:
            self.controller.close()
