"""Blocking adapter from vendored PyoMyo to the streamer device protocol."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from sifi_streamer.acquisition import SignalChannelSpec, SignalStreamSpec
from sifi_streamer.exceptions import DeviceError

MYO_EMG_SAMPLE_RATE = 200
MYO_IMU_SAMPLE_RATE = 50
MYO_EMG_STREAM_ID = "emg_armband"
MYO_IMU_STREAM_ID = "imu"
MYO_EMG_CHANNELS = tuple(f"emg{i}" for i in range(8))
MYO_IMU_CHANNELS = ("ax", "ay", "az", "qw", "qx", "qy", "qz", "gx", "gy", "gz")
MYO_STREAMS = (
    SignalStreamSpec(
        stream_id=MYO_EMG_STREAM_ID,
        channels=tuple(SignalChannelSpec(channel) for channel in MYO_EMG_CHANNELS),
        nominal_rate_hz=MYO_EMG_SAMPLE_RATE,
        label="Myo EMG",
    ),
    SignalStreamSpec(
        stream_id=MYO_IMU_STREAM_ID,
        channels=tuple(SignalChannelSpec(channel) for channel in MYO_IMU_CHANNELS),
        nominal_rate_hz=MYO_IMU_SAMPLE_RATE,
        label="Myo IMU",
    ),
)


MyoTransport = Literal["dongle", "ble"]


@dataclass(frozen=True, slots=True)
class MyoPacket:
    """Generic acquisition packet emitted by :class:`MyoDevice`."""

    stream_id: str
    timestamps: tuple[float, ...]
    data: Mapping[str, Sequence[float | int | None]]
    received_at: float
    reported_rate_hz: float | None = None
    samples_lost: int = 0
    status: str = "ok"

    def capture_document(self) -> dict[str, object]:
        """Return the schema-compatible raw packet stored in capture logs."""
        return {
            "packet_type": self.stream_id,
            "timestamps": list(self.timestamps),
            "data": {channel: list(values) for channel, values in self.data.items()},
            "received_at": self.received_at,
            "sample_rate": self.reported_rate_hz,
            "samples_lost": self.samples_lost,
            "status": self.status,
        }


class MyoDevice:
    """Read EMG and IMU packets from a Myo armband.

    Supports two transports: "dongle" talks BGAPI over the Bluegiga serial
    dongle (via the vendored PyoMyo), and "ble" connects directly to the
    armband's GATT services over the OS Bluetooth adapter (via bleak).
    """

    def __init__(
        self,
        tty: str | None = None,
        *,
        transport: MyoTransport = "dongle",
        address: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._tty = tty
        self._transport = transport
        self._address = address
        self._clock = clock
        self._myo: Any | None = None
        self._packets: deque[MyoPacket] = deque()
        self._packet_available = threading.Event()

    @property
    def streams(self) -> tuple[SignalStreamSpec, ...]:
        """Return the fixed generic stream registry for the Myo armband."""
        return MYO_STREAMS

    @property
    def device_info(self) -> dict[str, object]:
        if self._transport == "ble":
            return {"device": "Myo", "transport": "ble", "address": self._address}
        return {"device": "Myo", "transport": "serial", "tty": self._tty}

    def connect(self) -> None:
        if self._myo is not None:
            return
        try:
            from .pyomyo import emg_mode

            if self._transport == "ble":
                from .ble import BleMyo

                myo = BleMyo(address=self._address, mode=emg_mode.FILTERED)
            else:
                from .pyomyo import Myo

                myo = Myo(mode=emg_mode.FILTERED, tty=self._tty)
            myo.add_emg_handler(self._on_emg)
            myo.add_imu_handler(self._on_imu)
            myo.connect()
        except Exception as exc:
            raise DeviceError(f"MyoDevice: failed to connect: {exc}") from exc
        self._myo = myo

    def disconnect(self) -> None:
        myo, self._myo = self._myo, None
        if myo is not None:
            myo.disconnect()

    def read_packet(self) -> MyoPacket:
        if self._myo is None:
            raise DeviceError("MyoDevice.read_packet() called before connect()")
        try:
            while not self._packets:
                if self._transport == "ble":
                    self._packet_available.wait(timeout=1.0)
                    self._packet_available.clear()
                else:
                    self._myo.run()
        except Exception as exc:
            raise DeviceError(f"MyoDevice: serial read failed: {exc}") from exc
        return self._packets.popleft()

    def _on_emg(self, emg: Sequence[int], _moving: int) -> None:
        channels = MYO_EMG_CHANNELS
        if len(emg) != len(channels):
            return
        now = self._clock()
        self._packets.append(
            MyoPacket(
                stream_id=MYO_EMG_STREAM_ID,
                timestamps=(now,),
                data={
                    channel: [float(value)] for channel, value in zip(channels, emg, strict=True)
                },
                received_at=now,
                reported_rate_hz=float(MYO_EMG_SAMPLE_RATE),
            )
        )
        self._packet_available.set()

    def _on_imu(
        self,
        quaternion: Sequence[int],
        accelerometer: Sequence[int],
        gyroscope: Sequence[int],
    ) -> None:
        if not (len(quaternion) == 4 and len(accelerometer) == len(gyroscope) == 3):
            return
        now = self._clock()
        channels = MYO_IMU_CHANNELS
        values = (*accelerometer, *quaternion, *gyroscope)
        self._packets.append(
            MyoPacket(
                stream_id=MYO_IMU_STREAM_ID,
                timestamps=(now,),
                data={
                    channel: [float(value)] for channel, value in zip(channels, values, strict=True)
                },
                received_at=now,
                reported_rate_hz=float(MYO_IMU_SAMPLE_RATE),
            )
        )
        self._packet_available.set()
