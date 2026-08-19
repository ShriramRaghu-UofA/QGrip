"""Composable acquisition adapters and readiness checks."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from sifi_streamer.acquisition import AcquisitionDevice as StreamerAcquisitionDevice
from sifi_streamer.sifi import SiFiBandDevice

from qgrip.domain import DeviceConfig
from qgrip.errors import DeviceError


@dataclass(frozen=True, slots=True)
class SignalPacket:
    timestamp: float
    samples: tuple[tuple[float, ...], ...]
    sample_rate_hz: float


def _numeric_sample(value: float | int | None) -> float:
    if value is None:
        raise DeviceError("device packet contains a missing signal sample")
    return float(value)


@runtime_checkable
class SignalDevice(Protocol):
    @property
    def sample_rate_hz(self) -> float: ...

    @property
    def channels(self) -> int: ...

    def connect(self) -> None: ...

    def set_cue(self, gesture: str, activation: float) -> None: ...

    def read(self, count: int) -> SignalPacket: ...

    def close(self) -> None: ...


class SyntheticDevice:
    """Deterministic device used by tests, demos, and release smoke tests."""

    def __init__(self, config: DeviceConfig) -> None:
        self._config = config
        self._connected = False
        self._offset = 0
        self._random = np.random.default_rng(config.seed)
        self._gesture_index = 0
        self._activation = 0.0

    @property
    def sample_rate_hz(self) -> float:
        return self._config.sample_rate_hz

    @property
    def channels(self) -> int:
        return self._config.channels

    def connect(self) -> None:
        self._connected = True

    def set_cue(self, gesture: str, activation: float) -> None:
        """Select a reproducible class-specific waveform for synthetic workflows."""
        self._gesture_index = sum((index + 1) * ord(value) for index, value in enumerate(gesture))
        self._activation = activation

    def read(self, count: int) -> SignalPacket:
        if not self._connected:
            raise DeviceError("synthetic device is not connected")
        indices = np.arange(self._offset, self._offset + count, dtype=float)
        waves = []
        for channel in range(self.channels):
            frequency = 5 + self._gesture_index % 17 + channel * 0.5
            amplitude = 0.15 + 0.85 * self._activation
            signal = amplitude * np.sin(
                2 * math.pi * frequency * indices / self.sample_rate_hz + channel * 0.2
            )
            waves.append(signal + self._random.normal(0, 0.015, count))
        self._offset += count
        return SignalPacket(
            time.time(),
            tuple(tuple(float(item) for item in row) for row in np.stack(waves, axis=1)),
            self.sample_rate_hz,
        )

    def close(self) -> None:
        self._connected = False


class MyoDeviceAdapter:
    """Adapter around the attributed vendored PyoMyo/BLE implementation."""

    def __init__(self, config: DeviceConfig) -> None:
        self._config = config
        self._device: StreamerAcquisitionDevice | None = None

    @property
    def sample_rate_hz(self) -> float:
        return 200.0

    @property
    def channels(self) -> int:
        return 8

    def connect(self) -> None:
        try:
            from qgrip.vendor.myo.device import MyoDevice

            self._device = MyoDevice(
                transport="ble" if self._config.kind == "myo_ble" else "dongle",
                address=self._config.address,
                tty=self._config.port,
            )
            self._device.connect()
        except Exception as exc:
            self._device = None
            raise DeviceError(f"Myo connection failed: {exc}") from exc

    def set_cue(self, gesture: str, activation: float) -> None:
        """Physical devices observe cues but do not synthesize their signal."""

    def read(self, count: int) -> SignalPacket:
        if self._device is None:
            raise DeviceError("Myo device is not connected")
        rows: list[tuple[float, ...]] = []
        timestamp = time.time()
        while len(rows) < count:
            packet = self._device.read_packet()
            if getattr(packet, "stream_id", "") != "emg_armband":
                continue
            data = packet.data
            rows.append(tuple(_numeric_sample(data[f"emg{index}"][0]) for index in range(8)))
            if len(packet.timestamps):
                timestamp = float(packet.timestamps[-1])
        return SignalPacket(timestamp, tuple(rows), self.sample_rate_hz)

    def close(self) -> None:
        if self._device is not None:
            self._device.disconnect()
            self._device = None


class SiFiDeviceAdapter:
    """Late-bound adapter using only sifi-streamer's public surface."""

    def __init__(self, config: DeviceConfig) -> None:
        self._config = config
        self._device: StreamerAcquisitionDevice | None = None

    @property
    def sample_rate_hz(self) -> float:
        return self._config.sample_rate_hz

    @property
    def channels(self) -> int:
        return self._config.channels

    def connect(self) -> None:
        try:
            port = int(self._config.port) if self._config.port is not None else 5000
            self._device = SiFiBandDevice(host=self._config.address or "127.0.0.1", port=port)
            self._device.connect()
        except DeviceError:
            raise
        except Exception as exc:
            raise DeviceError(f"SiFi connection failed: {exc}") from exc

    def set_cue(self, gesture: str, activation: float) -> None:
        """Physical devices observe cues but do not synthesize their signal."""

    def read(self, count: int) -> SignalPacket:
        if self._device is None:
            raise DeviceError("SiFi device is not connected")
        packet = self._device.read_packet()
        data = packet.data
        if not isinstance(data, Mapping):
            raise DeviceError("unexpected SiFi packet shape")
        channels = list(data.values())[: self.channels]
        if not channels:
            raise DeviceError("SiFi packet contains no signal channels")
        rows = tuple(
            tuple(_numeric_sample(channels[column][row]) for column in range(len(channels)))
            for row in range(min(count, len(channels[0])))
        )
        return SignalPacket(time.time(), rows, self.sample_rate_hz)

    def close(self) -> None:
        if self._device is not None:
            self._device.disconnect()
            self._device = None


def create_device(config: DeviceConfig) -> SignalDevice:
    if config.kind == "synthetic":
        return SyntheticDevice(config)
    if config.kind in {"myo_ble", "myo_dongle"}:
        return MyoDeviceAdapter(config)
    if config.kind == "sifi":
        return SiFiDeviceAdapter(config)
    raise DeviceError(f"unsupported device: {config.kind}")


def check_device(config: DeviceConfig) -> dict[str, object]:
    device = create_device(config)
    try:
        device.connect()
        packet = device.read(max(1, min(8, int(device.sample_rate_hz / 20))))
        return {
            "ready": True,
            "kind": config.kind,
            "sample_rate_hz": device.sample_rate_hz,
            "channels": device.channels,
            "samples": len(packet.samples),
        }
    finally:
        device.close()
