"""Direct BLE backend for the Myo armband, talking GATT over the OS Bluetooth
adapter instead of the Bluegiga serial dongle.

Bridges bleak's asyncio client onto a dedicated background thread with its
own event loop, so callers get the same blocking connect()/run() shape as
the dongle backend in pyomyo.py.
"""

from __future__ import annotations

import asyncio
import struct
import threading
from collections.abc import Callable, Sequence
from typing import Any

from .pyomyo import emg_mode

CONTROL_SERVICE_UUID = "d5060001-a904-deb9-4748-2c7f4a124842"
MYO_INFO_CHAR_UUID = "d5060101-a904-deb9-4748-2c7f4a124842"
COMMAND_CHAR_UUID = "d5060401-a904-deb9-4748-2c7f4a124842"

IMU_DATA_SERVICE_UUID = "d5060002-a904-deb9-4748-2c7f4a124842"
IMU_DATA_CHAR_UUID = "d5060402-a904-deb9-4748-2c7f4a124842"

CLASSIFIER_EVENT_CHAR_UUID = "d5060103-a904-deb9-4748-2c7f4a124842"

EMG_DATA_SERVICE_UUID = "d5060005-a904-deb9-4748-2c7f4a124842"
EMG_DATA_CHAR_UUIDS = (
    "d5060105-a904-deb9-4748-2c7f4a124842",
    "d5060205-a904-deb9-4748-2c7f4a124842",
    "d5060305-a904-deb9-4748-2c7f4a124842",
    "d5060405-a904-deb9-4748-2c7f4a124842",
)

BATTERY_LEVEL_CHAR_UUID = "00002a19-0000-1000-8000-00805f9b34fb"

_EMG_MODE_COMMANDS = {
    emg_mode.FILTERED: b"\x01\x03\x02\x01\x01",
    emg_mode.RAW: b"\x01\x03\x03\x01\x00",
}


class BleMyo:
    """Connects to a Myo armband directly over BLE using bleak.

    Mirrors the subset of the pyomyo.Myo interface that MyoDevice depends
    on (add_emg_handler, add_imu_handler, connect, run, disconnect), so it
    can be swapped in without changing the device adapter's control flow.
    """

    def __init__(
        self,
        address: str | None = None,
        mode: emg_mode = emg_mode.FILTERED,
        *,
        scan_timeout: float = 10.0,
    ) -> None:
        self.address = address
        self.mode = mode
        self.scan_timeout = scan_timeout
        self.emg_handlers: list[Callable[[Sequence[int], int], None]] = []
        self.imu_handlers: list[Callable[[Sequence[int], Sequence[int], Sequence[int]], None]] = []

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client: Any | None = None

    def add_emg_handler(self, handler: Callable[[Sequence[int], int], None]) -> None:
        self.emg_handlers.append(handler)

    def add_imu_handler(
        self,
        handler: Callable[[Sequence[int], Sequence[int], Sequence[int]], None],
    ) -> None:
        self.imu_handlers.append(handler)

    def connect(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="myo-ble", daemon=True)
        self._thread.start()
        self._run_coroutine(self._connect_async()).result()

    def run(self) -> None:
        """No-op: bleak delivers notifications on the background loop."""

    def disconnect(self) -> None:
        if self._loop is None:
            return
        try:
            self._run_coroutine(self._disconnect_async()).result(timeout=5)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=5)
            self._loop.close()
            self._loop = None
            self._thread = None

    def _run_coroutine(self, coro: Any) -> Any:
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _connect_async(self) -> None:
        from bleak import BleakClient

        address = self.address
        if address is None:
            address = await self._discover_address()

        client = BleakClient(address)
        await client.connect(timeout=self.scan_timeout)
        self._client = client

        await client.start_notify(EMG_DATA_CHAR_UUIDS[0], self._on_emg_notify)
        await client.start_notify(EMG_DATA_CHAR_UUIDS[1], self._on_emg_notify)
        await client.start_notify(EMG_DATA_CHAR_UUIDS[2], self._on_emg_notify)
        await client.start_notify(EMG_DATA_CHAR_UUIDS[3], self._on_emg_notify)
        await client.start_notify(IMU_DATA_CHAR_UUID, self._on_imu_notify)

        command = _EMG_MODE_COMMANDS.get(self.mode)
        if command is not None:
            await client.write_gatt_char(COMMAND_CHAR_UUID, command, response=True)
        # Never sleep / disconnect while idle.
        await client.write_gatt_char(COMMAND_CHAR_UUID, b"\x09\x01\x01", response=True)

    async def _discover_address(self) -> str:
        from bleak import BleakScanner

        def is_myo(device: Any, adv: Any) -> bool:
            uuids = [uuid.lower() for uuid in adv.service_uuids]
            if CONTROL_SERVICE_UUID in uuids or EMG_DATA_SERVICE_UUID in uuids:
                return True
            return device.name is not None and "myo" in device.name.lower()

        device = await BleakScanner.find_device_by_filter(is_myo, timeout=self.scan_timeout)
        if device is None:
            raise RuntimeError(
                "No Myo armband found via BLE scan (advertises neither the Myo "
                "control/EMG service UUIDs nor a name containing 'myo'; pass "
                "--myo-address explicitly if the armband uses a custom name)"
            )
        return device.address

    async def _disconnect_async(self) -> None:
        client, self._client = self._client, None
        if client is not None and client.is_connected:
            await client.disconnect()

    def _on_emg_notify(self, _sender: Any, data: bytearray) -> None:
        emg1 = struct.unpack("<8b", bytes(data[:8]))
        emg2 = struct.unpack("<8b", bytes(data[8:16]))
        for handler in self.emg_handlers:
            handler(emg1, 0)
            handler(emg2, 0)

    def _on_imu_notify(self, _sender: Any, data: bytearray) -> None:
        vals = struct.unpack("<10h", bytes(data[:20]))
        quat = vals[:4]
        acc = vals[4:7]
        gyro = vals[7:10]
        for handler in self.imu_handlers:
            handler(quat, acc, gyro)
