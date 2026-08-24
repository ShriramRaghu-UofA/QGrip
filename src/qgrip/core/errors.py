"""Stable domain errors shared by command and HTTP adapters."""


class QGripError(Exception):
    """Base error safe to expose to users."""


class ValidationError(QGripError):
    """Input failed domain validation."""


class BusyError(QGripError):
    """A hardware-owning workflow is already active."""


class ArtifactError(QGripError):
    """An artifact is missing, malformed, or incompatible."""


class DeviceError(QGripError):
    """A device could not be prepared or read."""


class RpcError(QGripError):
    """Arduino Router RPC failed."""
