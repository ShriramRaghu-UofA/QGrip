"""Concurrent MessagePack-RPC client for the UNO Q Arduino Router socket."""

from __future__ import annotations

import socket
import threading
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Self

from qgrip.core.errors import RpcError

REQUEST = 0
RESPONSE = 1
NOTIFY = 2


@dataclass(slots=True)
class _Pending:
    """One in-flight request's completion signal and eventual response payload."""

    event: threading.Event = field(default_factory=threading.Event)
    result: object = None
    error: object = None


class MessagePackRpcClient:
    """Thread-safe Router client using streaming MessagePack-RPC array frames."""

    def __init__(
        self, socket_path: str = "/var/run/arduino-router.sock", timeout: float = 5.0
    ) -> None:
        """Initialize an unconnected client for one Router Unix-domain socket."""
        self.socket_path = socket_path
        self.timeout = timeout
        self._socket: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: dict[int, _Pending] = {}
        self._sequence = 0
        self._running = threading.Event()
        self._receiver: threading.Thread | None = None

    def connect(self) -> None:
        """Connect once and start the receiver that resolves concurrent calls."""
        if self._socket is not None:
            return
        try:
            family = getattr(socket, "AF_UNIX", None)
            if family is None:
                raise RpcError("Unix-domain sockets are unavailable on this platform")
            connection = socket.socket(family, socket.SOCK_STREAM)
            connection.connect(self.socket_path)
        except OSError as exc:
            raise RpcError(
                f"cannot connect to Arduino Router at {self.socket_path}: {exc}"
            ) from exc
        self._socket = connection
        self._running.set()
        self._receiver = threading.Thread(
            target=self._receive_loop, name="qgrip-router-rpc", daemon=True
        )
        self._receiver.start()

    def close(self) -> None:
        """Close transport and unblock all waiters with a disconnection error."""
        self._running.clear()
        connection, self._socket = self._socket, None
        if connection is not None:
            connection.close()
        with self._pending_lock:
            pending, self._pending = self._pending, {}
        for call in pending.values():
            call.error = "Arduino Router disconnected"
            call.event.set()
        if self._receiver is not None and self._receiver is not threading.current_thread():
            self._receiver.join(timeout=1)
        self._receiver = None

    def __enter__(self) -> Self:
        """Connect the client for use as a context manager."""
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the client when its context exits, regardless of the exception."""
        self.close()

    def call(self, method: str, *args: object) -> object:
        """Send a request and wait up to the configured timeout for its response."""
        try:
            import msgpack
        except ImportError as exc:
            raise RpcError("install QGrip with the 'handi' extra for MessagePack RPC") from exc
        connection = self._socket
        if connection is None:
            raise RpcError("RPC client is not connected")
        with self._pending_lock:
            self._sequence += 1
            sequence = self._sequence
            pending = _Pending()
            self._pending[sequence] = pending
        payload = msgpack.packb([REQUEST, sequence, method, list(args)], use_bin_type=True)
        try:
            with self._send_lock:
                connection.sendall(payload)
        except OSError as exc:
            with self._pending_lock:
                self._pending.pop(sequence, None)
            raise RpcError(f"RPC send failed: {exc}") from exc
        if not pending.event.wait(self.timeout):
            with self._pending_lock:
                self._pending.pop(sequence, None)
            raise RpcError(f"timeout waiting for {method!r}")
        if pending.error:
            raise RpcError(str(pending.error))
        return pending.result

    def notify(self, method: str, *args: object) -> None:
        """Send a fire-and-forget MessagePack-RPC notification."""
        import msgpack

        connection = self._socket
        if connection is None:
            raise RpcError("RPC client is not connected")
        with self._send_lock:
            connection.sendall(msgpack.packb([NOTIFY, method, list(args)], use_bin_type=True))

    def _receive_loop(self) -> None:
        """Decode streamed frames and release pending requests until disconnect."""
        import msgpack

        unpacker = msgpack.Unpacker(raw=False)
        connection = self._socket
        assert connection is not None
        try:
            while self._running.is_set():
                data = connection.recv(4096)
                if not data:
                    raise RpcError("Arduino Router disconnected")
                unpacker.feed(data)
                for message in unpacker:
                    self._handle(message)
        except (OSError, RpcError, ValueError) as exc:
            with self._pending_lock:
                pending, self._pending = self._pending, {}
            for call in pending.values():
                call.error = str(exc)
                call.event.set()
            self._running.clear()

    def _handle(self, message: Any) -> None:
        """Route one valid response frame to the matching pending request."""
        if not isinstance(message, list) or len(message) < 4 or message[0] != RESPONSE:
            return
        sequence = message[1]
        if not isinstance(sequence, int):
            return
        with self._pending_lock:
            pending = self._pending.pop(sequence, None)
        if pending is not None:
            pending.error = message[2]
            pending.result = message[3]
            pending.event.set()
