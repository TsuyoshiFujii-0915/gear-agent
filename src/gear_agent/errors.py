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


def safe_error_payload(error: GearError, secrets: tuple[str, ...]) -> dict[str, str]:
    """Projects an error for diagnostics without copying arbitrary details.

    Args:
        error: Original structured failure.
        secrets: Nonempty credentials and endpoint URLs to redact.

    Returns:
        Error identity and message with configured secrets removed.
    """
    payload = {'type': error.error_type, 'origin': error.origin, 'message': error.message}
    for secret in sorted(secrets, key=len, reverse=True):
        for key, value in payload.items():
            payload[key] = value.replace(secret, '[REDACTED]')
    return payload
