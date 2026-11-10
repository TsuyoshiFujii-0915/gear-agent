from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os
import stat

from gear_agent.store.base import ContextStore


class JsonlContextStore(ContextStore):
    """JSONL-backed context store."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def append(self, session_id: str, kind: str, payload: dict[str, Any]) -> None:
        event = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "kind": kind,
            "payload": payload,
        }
        path = self._path_for_session(session_id)
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(event, ensure_ascii=False))
            output.write("\n")

    def append_request(self, session_id: str, payload: dict[str, Any]) -> None:
        """Appends one content-free JSON line outside canonical session discovery."""
        try:
            import fcntl
        except ImportError as error:
            raise OSError('Request log needs advisory file locking.') from error
        path = self._path_for_session(session_id).with_suffix('.requests.ndjson')
        line = (json.dumps(payload, allow_nan=False) + '\n').encode('utf-8')
        flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK
        fd = os.open(path, flags, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise OSError('Request log destination must be a single regular file.')
            if info.st_size and os.pread(fd, 1, info.st_size - 1) != b'\n':
                raise ValueError('Request log ends with a partial record.')
            if os.write(fd, line) != len(line):
                os.ftruncate(fd, info.st_size)
                raise OSError('Incomplete request log append.')
        finally:
            os.close(fd)

    def agent_requests(self, session_id: str) -> Iterator[dict[str, Any]]:
        """Reads this session's terminal agent observations in append order.

        Args:
            session_id: Session whose request sidecar should be inspected.

        Yields:
            Schema 1 agent records from the session sidecar.

        Raises:
            OSError: If the log cannot safely be read.
            ValueError: If a record is incomplete or has an unsupported schema.
        """
        path = self._path_for_session(session_id).with_suffix(".requests.ndjson")
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return
        except AttributeError as error:
            raise OSError("Safe request log access is unsupported on this platform.") from error
        with os.fdopen(fd, "rb") as source:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise OSError("Request log must be a single regular file.")
            for number, line in enumerate(source, 1):
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeError) as error:
                    raise ValueError(f"Invalid JSON in request log at line {number}.") from error
                if not line.endswith(b"\n") or not isinstance(row, dict) or row.get("schema_version") != 1:
                    raise ValueError(f"Incomplete or unsupported request log record at line {number}.")
                if row.get("session_id") != session_id or row.get("purpose") != "agent":
                    continue
                for key in ("request_id", "started_at", "model_identity", "endpoint_identity"):
                    if not isinstance(row.get(key), str) or not row[key]:
                        raise ValueError(f"Invalid {key} in request log at line {number}.")
                yield row

    def load(self, session_id: str) -> list[dict[str, Any]]:
        path = self._path_for_session(session_id)
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                events.append(parsed)
        return events

    def _path_for_session(self, session_id: str) -> Path:
        safe_session_id = session_id.replace("/", "_")
        return self._root / f"{safe_session_id}.jsonl"
