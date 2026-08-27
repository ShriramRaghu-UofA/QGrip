"""Standalone EMG-inference runtime that drives a USB HID joystick axis.

This is the :mod:`qgrip.runtime.handi` runtime with a different output stage.
Acquisition, model loading, the windowed inference loop, the confidence gate,
and the :class:`~qgrip.capture.streaming.PredictionDebouncer` are all identical
to Handi; the only difference is that an accepted prediction is mapped to a
signed deflection on one axis of a USB HID game controller instead of a clamped
joint pose sent over the Router RPC socket.

Each accepted prediction is written to a HID gadget device (default
``/dev/hidg1``) as a 4-byte report ``[X, Y, Z, buttons]``. ``LABEL_TO_AXIS``
gives every mapped gesture its own axis (``open`` -> X, ``close`` -> Y); Z and
the buttons are unused, and ``rest`` (or any unmapped label) sends the neutral
all-zero report that recentres the stick.

The report is re-sent every ``inference.inference_period_seconds`` tick for as
long as the same gesture keeps winning, so a sustained contraction reads as a
held axis deflection rather than a single tap. On stop -- Ctrl-C, SIGTERM, or
any error -- the neutral report is written once more before the device closes.

The deflection magnitude is scaled by the prediction's proportional
``activation`` between zero and ``AXIS_MAX``; ``AXIS_MAX`` itself is a hardcoded
placeholder for a per-gesture calibrated maximum.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import BinaryIO

from qgrip.capture.streaming import LiveEMGSession, PredictionDebouncer, sample_rates_match
from qgrip.core.domain import Prediction, QGripProfile
from qgrip.core.errors import DeviceError, ValidationError
from qgrip.runtime.workflows import InferenceService

LOGGER = logging.getLogger("qgrip.runtime.hid")

#: Placeholder full-scale axis deflection, matching the HID descriptor's signed
#: Logical Maximum (127) / Minimum (-127). Replace with a per-gesture calibrated
#: value once the classifier reports a measured amplitude instead of a class.
AXIS_MAX = 127
AXIS_MIN = -127

#: Wire format written to the HID gadget: X, Y, Z, buttons (one byte each).
HIDG_REPORT_LENGTH = 4
NEUTRAL_REPORT = bytes(HIDG_REPORT_LENGTH)
AXIS_NAMES = ("X", "Y", "Z")

#: Which axis each gesture drives, ``label -> (axis index, signed value)``.
#: Labels absent here -- including ``rest`` -- fall back to the neutral report.
#: Keys are model labels; edit this to remap axes.
LABEL_TO_AXIS: dict[str, tuple[int, int]] = {
    "open": (0, AXIS_MAX),
    "close": (1, AXIS_MAX),
    "wrist_flexion": (2, AXIS_MAX),
    "wrist_extension": (2, AXIS_MIN),
}


def build_report(label: str, activation: float = 1.0) -> bytes:
    """Return the 4-byte joystick report for one label.

    Unmapped labels (and ``rest``) return the neutral all-zero report. For a
    mapped label the configured axis is set to ``round(value * activation)``
    with ``activation`` clamped to ``[0, 1]``; the byte is written two's
    complement (``& 0xFF``, so ``-127`` becomes ``0x81``) to match the HID
    descriptor's signed Logical Minimum/Maximum range.
    """
    mapping = LABEL_TO_AXIS.get(label)
    if mapping is None:
        return NEUTRAL_REPORT
    axis, value = mapping
    scaled = round(value * min(max(activation, 0.0), 1.0))
    axes = [0, 0, 0]
    axes[axis] = scaled
    return bytes(component & 0xFF for component in axes) + b"\x00"


class JoystickController:
    """Owns the HID gadget device and maps predictions to held axis deflections."""

    def __init__(self, device: Path = Path("/dev/hidg1")) -> None:
        """Retain the target device node; do not open it until ``connect``."""
        self.device = device
        self._lock = threading.RLock()
        self._hidg: BinaryIO | None = None
        self._last_report = NEUTRAL_REPORT
        self._last_label = "rest"

    @property
    def last_report(self) -> bytes:
        """Return the most recently built report under the controller lock."""
        with self._lock:
            return self._last_report

    def connect(self) -> None:
        """Open the HID gadget device for unbuffered binary writes."""
        if not self.device.exists():
            raise DeviceError(
                f"no HID gadget device at {self.device} -- is the joystick function configured?"
            )
        # buffering=0: every write(2) must land as exactly one interrupt-IN
        # report; a buffered stream could coalesce or split reports.
        self._hidg = open(self.device, "wb", buffering=0)  # noqa: SIM115
        LOGGER.info("HID reports -> %s", self.device)

    def _write(self, report: bytes) -> None:
        """Send one raw report, requiring the device to be open."""
        if self._hidg is None:
            raise DeviceError("HID gadget device is not open")
        self._hidg.write(report)

    def apply_neutral(self) -> None:
        """Recentre every axis by sending (and latching) the neutral report."""
        with self._lock:
            self._last_report = NEUTRAL_REPORT
            self._last_label = "rest"
        self._write(NEUTRAL_REPORT)

    def apply_prediction(self, prediction: Prediction) -> None:
        """Map an accepted prediction to an axis report, latch it, and send it."""
        activation = 0.0 if prediction.gesture == "rest" else prediction.activation
        report = build_report(prediction.gesture, activation)
        with self._lock:
            self._last_report = report
            self._last_label = prediction.gesture
        self._write(report)

    def resend(self) -> None:
        """Re-send the latched report so a sustained gesture holds its deflection."""
        self._write(self.last_report)

    def close(self) -> None:
        """Recentre the joystick and close the HID gadget device exactly once."""
        with self._lock:
            hidg = self._hidg
            self._hidg = None
            self._last_report = NEUTRAL_REPORT
            self._last_label = "rest"
        if hidg is not None:
            try:
                hidg.write(NEUTRAL_REPORT)
            finally:
                hidg.close()
            LOGGER.info("stopped -- HID report cleared")


class HidRuntime:
    """Owns acquisition, inference, the HID joystick, and exactly-once cleanup."""

    def __init__(
        self,
        profile: QGripProfile,
        model: str,
        *,
        device: Path = Path("/dev/hidg1"),
    ) -> None:
        """Construct the runtime and load the checkpoint; open no hardware yet."""
        self.profile = profile
        self.model = InferenceService(
            model, profile.inference.backend, profile.inference.device_preference
        )
        self.controller = JoystickController(device)
        self._stop = threading.Event()
        self._closed = False
        self._close_lock = threading.Lock()

    def validate(self) -> None:
        """Verify the profile's device identity against the checkpoint metadata."""
        metadata = self.model.metadata
        if self.model.channels != self.profile.device.channels:
            raise ValidationError("model channel count does not match device")
        if not sample_rates_match(
            float(metadata["sample_rate_hz"]), self.profile.device.sample_rate_hz
        ):
            raise ValidationError("model sample rate does not match device")

    def start(self) -> None:
        """Validate configuration, open the HID device, and recentre the stick."""
        self.validate()
        try:
            self.controller.connect()
            self.controller.apply_neutral()
        except BaseException:
            self.close()
            raise

    def run(self) -> None:
        """Own live acquisition/inference/output until stopped or an error occurs."""
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
                resend_period = self.profile.inference.inference_period_seconds
                last_resend_at = 0.0
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
                    # Re-send on a steady cadence, not just on change -- this is
                    # what makes a sustained gesture read as a held axis
                    # deflection rather than a single tap.
                    now = time.monotonic()
                    if now - last_resend_at >= resend_period:
                        self.controller.resend()
                        last_resend_at = now
        finally:
            self.close()

    def stop(self) -> None:
        """Request cooperative exit from the live inference loop."""
        self._stop.set()

    def close(self) -> None:
        """Perform exactly-once runtime cleanup, recentring the joystick."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        self.controller.close()


def main() -> int:
    """Expose the HID-joystick runtime as a console script without a separate module."""
    import sys

    from qgrip.runtime.cli import main as qgrip_main

    return qgrip_main(["hid", *sys.argv[1:]])
