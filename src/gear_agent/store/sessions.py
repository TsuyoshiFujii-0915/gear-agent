from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gear_agent.errors import GearError, gear_error


@dataclass(frozen=True)
class PersistedSession:
    """Metadata for one persisted JSONL session.

    Attributes:
        session_id: Session identifier derived from the JSONL filename.
        updated_at_ns: File modification time in nanoseconds.
    """

    session_id: str
    updated_at_ns: int


class JsonlSessionDiscovery:
    """Discovers and resolves sessions persisted as JSONL files."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def resolve_session_id(self, reference: str) -> str:
        """Resolves a full session ID or unique session ID prefix.

        Args:
            reference: Full persisted session ID or unique prefix.

        Returns:
            Resolved full session ID.

        Raises:
            GearError: If no session matches or a prefix is ambiguous.
        """

        sessions = self.discover_sessions()
        for session in sessions:
            if session.session_id == reference:
                return session.session_id

        matches = [
            session.session_id
            for session in sessions
            if session.session_id.startswith(reference)
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise gear_error(
                "session_prefix_ambiguous",
                f"Session prefix '{reference}' is ambiguous: {', '.join(matches)}",
                "session_discovery",
                True,
                {
                    "reference": reference,
                    "session_dir": str(self._root),
                    "matching_session_ids": matches,
                },
            )
        raise gear_error(
            "session_not_found",
            f"No persisted session matches '{reference}'.",
            "session_discovery",
            True,
            {"reference": reference, "session_dir": str(self._root)},
        )

    def latest_session_id(self) -> str:
        """Returns the most recently updated persisted session ID.

        Modification-time ties are resolved by choosing the lexicographically
        greatest session ID.

        Returns:
            Resolved full session ID.

        Raises:
            GearError: If no persisted sessions exist.
        """

        sessions = self.discover_sessions()
        if len(sessions) == 0:
            raise gear_error(
                "latest_session_not_found",
                f"No persisted sessions exist in {self._root}.",
                "session_discovery",
                True,
                {"session_dir": str(self._root)},
            )
        latest = max(
            sessions,
            key=lambda session: (session.updated_at_ns, session.session_id),
        )
        return latest.session_id

    def discover_sessions(self) -> list[PersistedSession]:
        """Discovers persisted JSONL sessions and their update metadata.

        Returns:
            Sessions ordered by session ID.

        Raises:
            GearError: If the session directory cannot be scanned or inspected.
        """

        try:
            paths = sorted(self._root.glob("*.jsonl"))
            sessions: list[PersistedSession] = []
            for path in paths:
                if path.is_symlink():
                    raise _invalid_session_file(path, "Symbolic links are not supported.")
                if not path.is_file():
                    continue
                session_id = path.name.removesuffix(".jsonl")
                if session_id == "":
                    raise _invalid_session_file(path, "Session ID must not be empty.")
                sessions.append(
                    PersistedSession(
                        session_id=session_id,
                        updated_at_ns=path.stat().st_mtime_ns,
                    )
                )
        except OSError as exc:
            raise gear_error(
                "session_discovery_failed",
                f"Failed to inspect session directory: {self._root}",
                "session_discovery",
                True,
                {"session_dir": str(self._root), "reason": str(exc)},
            ) from exc
        return sessions


def _invalid_session_file(path: Path, reason: str) -> GearError:
    return gear_error(
        "session_file_invalid",
        f"Invalid persisted session file: {path}. {reason}",
        "session_discovery",
        True,
        {"path": str(path), "reason": reason},
    )
