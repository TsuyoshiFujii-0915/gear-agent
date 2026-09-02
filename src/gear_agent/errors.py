from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GearError(Exception):
    """Application error with explicit origin and recovery metadata.

    Attributes:
        error_type: Stable category for programmatic handling.
        message: Human-readable error message.
        origin: Component that produced the error.
        recoverable: Whether user action can reasonably recover from the error.
        details: Non-secret contextual details.
    """

    error_type: str
    message: str
    origin: str
    recoverable: bool
    details: dict[str, object]

    def __str__(self) -> str:
        return f"{self.origin}: {self.message}"


def gear_error(
    error_type: str,
    message: str,
    origin: str,
    recoverable: bool,
    details: dict[str, object],
) -> GearError:
    """Builds a GearError without hiding required metadata.

    Args:
        error_type: Stable category for programmatic handling.
        message: Human-readable error message.
        origin: Component that produced the error.
        recoverable: Whether user action can reasonably recover from the error.
        details: Non-secret contextual details.

    Returns:
        A populated GearError instance.
    """

    return GearError(error_type, message, origin, recoverable, details)
