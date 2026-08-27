"""sifi-streamer composition for QGrip devices and live EMG consumers."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from functools import partial
from math import isclose
from typing import Any, cast

import numpy as np
from sifi_streamer.acquisition import BackgroundHandle, StreamerConfig
from sifi_streamer.acquisition.devices import DeviceFactory, SignalChannelSpec, SignalStreamSpec
from sifi_streamer.acquisition.health import HealthThresholds
from sifi_streamer.acquisition.runtime import AcquisitionMonitor
from sifi_streamer.sifi import BridgeTransport, SiFiBridgeDevice, SyntheticSiFiDevice
from sifi_streamer.sifi.sensor_profile import EMG_IMU_PROFILE

from qgrip.core.domain import (
    AcquisitionConfig,
    DeviceConfig,
    DeviceKind,
    LiveSignalHealth,
    Prediction,
    SignalHealthSeverity,
)
from qgrip.core.errors import DeviceError

EMG_STREAM_ID = "emg_armband"


def sample_rates_match(left: float, right: float) -> bool:
    """Compare nominal hardware rates while tolerating harmless float encoding noise."""
    return isclose(left, right, rel_tol=0.0, abs_tol=0.1)


class PredictionDebouncer:
    """Accept a changed gesture only after consecutive inference outputs agree."""

    def __init__(self, required_predictions: int) -> None:
        """Require a positive count before a changed gesture can be accepted."""
        if required_predictions <= 0:
            raise ValueError("required predictions must be positive")
        self._required_predictions = required_predictions
        self._accepted_gesture: str | None = None
        self._candidate_gesture: str | None = None
        self._candidate_count = 0

    def accept(self, prediction: Prediction) -> Prediction | None:
        """Return stable predictions immediately and new gestures after agreement."""
        if prediction.gesture == self._accepted_gesture:
            self._candidate_gesture = None
            self._candidate_count = 0
            return prediction
        if prediction.gesture != self._candidate_gesture:
            self._candidate_gesture = prediction.gesture
            self._candidate_count = 0
        self._candidate_count += 1
        if self._candidate_count < self._required_predictions:
            return None
        self._accepted_gesture = prediction.gesture
        self._candidate_gesture = None
        self._candidate_count = 0
        return prediction


@dataclass(slots=True)
class MyoPacket:
    """Myo packet normalized to the public streamer acquisition protocol."""

    timestamp: float
    values: tuple[float, ...]
    sample_rate_hz: float

    @property
    def stream_id(self) -> str:
        """Return the public stream name consumed by QGrip and sifi-streamer."""
        return EMG_STREAM_ID

    @property
    def timestamps(self) -> tuple[float, ...]:
        """Expose this one-sample packet's acquisition timestamp as a stream tuple."""
        return (self.timestamp,)

    @property
    def data(self) -> dict[str, tuple[float, ...]]:
        """Expose ordered Myo values under the public ``emg<index>`` channel names."""
        return {f"emg{index}": (value,) for index, value in enumerate(self.values)}

    @property
    def reported_rate_hz(self) -> float:
        """Return the configured nominal source rate for health assessment."""
        return self.sample_rate_hz

    @property
    def samples_lost(self) -> int:
        """Report zero because this transport packet does not expose loss counts."""
        return 0

    @property
    def status(self) -> str:
        """Return the protocol's normal packet-status marker."""
        return "ok"

    def capture_document(self) -> dict[str, object]:
        """Convert the protocol packet to the durable capture-log wire document."""
        return {
            "packet_type": EMG_STREAM_ID,
            "timestamps": list(self.timestamps),
            "data": {name: list(values) for name, values in self.data.items()},
            "sample_rate": self.reported_rate_hz,
            "samples_lost": self.samples_lost,
            "status": self.status,
        }


class MyoAcquisitionDevice:
    """Inject the vendored Myo transport into sifi-streamer's device protocol."""

    def __init__(self, config: DeviceConfig) -> None:
        """Retain immutable device configuration until the worker connects."""
        self._config = config
        self._device: Any | None = None

    @property
    def streams(self) -> tuple[SignalStreamSpec, ...]:
        """Describe the fixed eight-channel EMG stream expected from Myo."""
        return (
            SignalStreamSpec(
                EMG_STREAM_ID,
                tuple(SignalChannelSpec(f"emg{index}") for index in range(8)),
                self._config.sample_rate_hz,
            ),
        )

    @property
    def device_info(self) -> dict[str, object]:
        """Return serializable Myo identity for capture provenance."""
        return {
            "device": "myo",
            "transport": self._config.kind,
            "sample_rate_hz": self._config.sample_rate_hz,
        }

    def connect(self) -> None:
        """Instantiate and connect the selected vendored Myo transport."""
        try:
            from qgrip.vendor.myo.device import MyoDevice

            self._device = MyoDevice(
                transport="ble" if self._config.kind == DeviceKind.MYO_BLE else "dongle",
                address=self._config.address,
                tty=self._config.port,
            )
            self._device.connect()
        except Exception as exc:
            self._device = None
            raise DeviceError(f"Myo connection failed: {exc}") from exc

    def disconnect(self) -> None:
        """Release a connected Myo transport, if startup reached that stage."""
        if self._device is not None:
            self._device.disconnect()
            self._device = None

    def read_packet(self) -> MyoPacket:
        """Read one transport sample and normalize it to streamer packet protocol."""
        if self._device is None:
            raise DeviceError("Myo device is not connected")
        while True:
            packet = self._device.read_packet()
            if packet is None:
                time.sleep(0.001)
                continue
            if packet.stream_id != EMG_STREAM_ID:
                continue
            data = packet.data
            values = tuple(float(data[f"emg{index}"][0]) for index in range(8))
            timestamps = packet.timestamps
            timestamp = float(timestamps[-1]) if timestamps else 0.0
            return MyoPacket(timestamp, values, self._config.sample_rate_hz)


def streamer_config(config: AcquisitionConfig) -> StreamerConfig:
    """Adapt QGrip's profile section to the streamer's public configuration."""
    return StreamerConfig(
        ring_buffer_seconds=config.ring_buffer_seconds,
        ack_timeout_s=config.ack_timeout_seconds,
        capture_log_enabled=config.capture_log_enabled,
        capture_frame_target_bytes=config.capture_frame_target_bytes,
        capture_flush_interval_s=config.capture_flush_interval_seconds,
        capture_compression_level=config.capture_compression_level,
        capture_fsync_on_boundary=config.capture_fsync_on_boundary,
    )


def health_thresholds(config: AcquisitionConfig) -> HealthThresholds:
    """Translate QGrip health options to the public streamer threshold value."""
    health = config.health
    return HealthThresholds(
        window_seconds=health.window_seconds,
        stale_after_seconds=health.stale_after_seconds,
        minimum_rate_ratio=health.minimum_rate_ratio,
        maximum_rate_ratio=health.maximum_rate_ratio,
        maximum_missing_fraction=health.maximum_missing_fraction,
        maximum_lost_samples=health.maximum_lost_samples,
    )


def streamer_device_factory(config: DeviceConfig) -> DeviceFactory:
    """Return a picklable factory used only by a streamer-owned worker."""
    if config.kind == DeviceKind.SIFI:
        if config.imu_sample_rate_hz is None:
            raise DeviceError(f"device.imu_sample_rate_hz must be set for {config.kind}")
        profile = replace(
            EMG_IMU_PROFILE,
            emg=replace(EMG_IMU_PROFILE.emg, sample_rate_hz=round(config.sample_rate_hz)),
            imu=replace(EMG_IMU_PROFILE.imu, sample_rate_hz=round(config.imu_sample_rate_hz)),
        )
        return cast(
            DeviceFactory,
            partial(
                SiFiBridgeDevice,
                host=config.address or "127.0.0.1",
                port=int(config.port or "5000"),
                transport=BridgeTransport.TCP,
                sensor_profile=profile,
            ),
        )
    if config.kind == DeviceKind.SYNTHETIC:
        return cast(
            DeviceFactory,
            partial(SyntheticSiFiDevice, emg_sample_rate=round(config.sample_rate_hz)),
        )
    if config.kind in {DeviceKind.MYO_BLE, DeviceKind.MYO_DONGLE}:
        return cast(DeviceFactory, partial(MyoAcquisitionDevice, config))
    raise DeviceError(f"unsupported device: {config.kind}")


class LiveEMGSession:
    """One streamer-owned acquisition worker with a validated live EMG reader."""

    def __init__(self, config: DeviceConfig, acquisition: AcquisitionConfig | None = None) -> None:
        """Prepare an unopened live session and its absolute consumer cursor."""
        acquisition = acquisition or AcquisitionConfig()
        self._handle = BackgroundHandle(
            streamer_config(acquisition), streamer_device_factory(config)
        )
        self._acquisition = acquisition
        self._monitor: AcquisitionMonitor | None = None
        self._reader: Any | None = None
        self._stream = None
        self._cursor = 0
        self._samples: np.ndarray | None = None
        self._filled = 0
        self._last_inference_end = 0
        self._emitted = False
        self._consumer_overruns = 0
        self._health: LiveSignalHealth | None = None

    def __enter__(self) -> LiveEMGSession:
        """Start streamer acquisition and initialize a cursor at its current tail."""
        self._handle.__enter__()
        self._monitor = AcquisitionMonitor(self._handle, health_thresholds(self._acquisition))
        try:
            stream = next(item for item in self._handle.streams if item.stream_id == EMG_STREAM_ID)
        except StopIteration as exc:
            self._handle.__exit__(None, None, None)
            raise DeviceError("device does not expose an emg_armband stream") from exc
        self._stream = stream
        self._reader = self._handle.stream_readers[EMG_STREAM_ID]
        self._cursor = 0
        self._samples = None
        self._filled = 0
        self._last_inference_end = 0
        self._emitted = False
        self._consumer_overruns = 0
        self._health = LiveSignalHealth()
        return self

    def __exit__(self, *args: object) -> None:
        """Stop the session-owned background acquisition worker on context exit."""
        self._handle.__exit__(*args)

    @property
    def sample_rate_hz(self) -> float:
        """Return the nominal sample rate reported by the active streamer handle."""
        assert self._stream is not None
        return self._stream.nominal_rate_hz

    @property
    def channels(self) -> int:
        """Return the validated EMG channel count from the active stream spec."""
        assert self._stream is not None
        return self._stream.n_channels

    @property
    def health(self) -> LiveSignalHealth | None:
        """Latest streamer health plus QGrip's consumer-overrun count."""
        self._update_health()
        return self._health

    def _update_health(self) -> None:
        """Merge streamer health with QGrip consumer-overrun accounting."""
        snapshot = self._monitor.latest() if self._monitor else None
        if snapshot is None:
            return
        stream = next((item for item in snapshot.streams if item.stream_id == EMG_STREAM_ID), None)
        if stream is None:
            return
        warnings = list(stream.warnings)
        if self._consumer_overruns:
            warnings.append(f"consumer ring-buffer overruns: {self._consumer_overruns}")
        self._health = LiveSignalHealth(
            SignalHealthSeverity(stream.severity.value),
            tuple(warnings),
            sum(stream.missing_by_channel),
            stream.lost_samples,
            stream.malformed_packets,
            stream.misaligned_packets,
            self._consumer_overruns,
        )

    def next_window(self, size: int, minimum_new_samples: int) -> np.ndarray | None:
        """Return a valid rolling window after enough new EMG samples arrive.

        The cursor is absolute, so consumer timing does not determine which
        samples are considered fresh.  A gap or ring-buffer overrun resets the
        window instead of running inference across discontinuous signal data.
        """
        if size <= 0 or minimum_new_samples <= 0:
            raise ValueError("window size and minimum new samples must be positive")
        assert self._reader is not None
        fatal = self._monitor.fatal() if self._monitor else None
        if fatal is not None:
            raise DeviceError(f"acquisition worker failed: {fatal.message}")
        self._update_health()
        window = self._reader.read_since(self._cursor)
        if window is None:
            return None
        self._cursor = window.end_index
        if self._samples is None or self._samples.shape[0] != size:
            channels = window.samples.shape[1]
            self._samples = np.zeros((size, channels), dtype=np.float32)
            self._filled = 0
            self._last_inference_end = 0
            self._emitted = False
        if window.overrun or not np.all(window.validity):
            if window.overrun:
                self._consumer_overruns += 1
                if self._health is not None:
                    self._health = replace(
                        self._health,
                        warnings=(*self._health.warnings, "consumer ring-buffer overrun"),
                        consumer_overruns=self._consumer_overruns,
                    )
            self._filled = 0
            self._last_inference_end = window.end_index
            self._emitted = False
            self._update_health()
            return None
        new_rows = np.asarray(window.samples, dtype=np.float32)
        n_new = new_rows.shape[0]
        if n_new >= size:
            self._samples[:] = new_rows[-size:]
        else:
            self._samples[: size - n_new] = self._samples[n_new:]
            self._samples[size - n_new :] = new_rows
        self._filled = min(size, self._filled + n_new)
        if self._filled < size or (
            self._emitted and window.end_index - self._last_inference_end < minimum_new_samples
        ):
            return None
        self._last_inference_end = window.end_index
        self._emitted = True
        return self._samples.copy()


def check_streamer_device(
    config: DeviceConfig, acquisition: AcquisitionConfig | None = None
) -> dict[str, object]:
    """Probe a device through the same worker path used in production."""
    with LiveEMGSession(config, acquisition) as session:
        return {
            "ready": True,
            "kind": config.kind,
            "sample_rate_hz": session.sample_rate_hz,
            "channels": session.channels,
        }
