import json
import os
import tempfile
import unittest
from pathlib import Path

from gear_agent.errors import GearError
from gear_agent.store.jsonl import JsonlContextStore
from gear_agent.store.sessions import JsonlSessionDiscovery


class StoreTests(unittest.TestCase):
    def test_appends_jsonl_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JsonlContextStore(Path(temp_dir))

            store.append("session-1", "user_input", {"text": "hello"})

            events_path = Path(temp_dir) / "session-1.jsonl"
            lines = events_path.read_text(encoding="utf-8").splitlines()
            event = json.loads(lines[0])
            self.assertEqual(event["kind"], "user_input")
            self.assertEqual(event["payload"], {"text": "hello"})

    def test_full_session_id_match_wins_over_longer_prefix_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JsonlContextStore(root)
            store.append("session-1234", "user_input", {"text": "exact"})
            store.append("session-1234-extra", "user_input", {"text": "longer"})
            discovery = JsonlSessionDiscovery(root)

            session_id = discovery.resolve_session_id("session-1234")

            self.assertEqual(session_id, "session-1234")

    def test_unique_session_id_prefix_resolves_persisted_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JsonlContextStore(root)
            store.append(
                "12345678-1234-5678-1234-567812345678",
                "user_input",
                {"text": "hello"},
            )
            discovery = JsonlSessionDiscovery(root)

            session_id = discovery.resolve_session_id("12345678")

            self.assertEqual(session_id, "12345678-1234-5678-1234-567812345678")

    def test_ambiguous_session_prefix_reports_matching_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JsonlContextStore(root)
            matching_ids = [
                "12345678-1111-1111-1111-111111111111",
                "12345678-2222-2222-2222-222222222222",
            ]
            for session_id in matching_ids:
                store.append(session_id, "user_input", {"text": session_id})
            discovery = JsonlSessionDiscovery(root)

            with self.assertRaises(GearError) as raised:
                discovery.resolve_session_id("12345678")

            self.assertEqual(raised.exception.error_type, "session_prefix_ambiguous")
            self.assertEqual(raised.exception.origin, "session_discovery")
            self.assertTrue(raised.exception.recoverable)
            self.assertEqual(
                raised.exception.details["matching_session_ids"],
                matching_ids,
            )
            for session_id in matching_ids:
                self.assertIn(session_id, str(raised.exception))

    def test_unknown_session_reports_error_without_creating_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            JsonlContextStore(root)
            discovery = JsonlSessionDiscovery(root)

            with self.assertRaises(GearError) as raised:
                discovery.resolve_session_id("unknown")

            self.assertEqual(raised.exception.error_type, "session_not_found")
            self.assertTrue(raised.exception.recoverable)
            self.assertEqual(list(root.glob("*.jsonl")), [])

    def test_latest_session_resolves_only_persisted_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JsonlContextStore(root)
            store.append("session-only", "user_input", {"text": "hello"})
            discovery = JsonlSessionDiscovery(root)

            session_id = discovery.latest_session_id()

            self.assertEqual(session_id, "session-only")

    def test_latest_session_uses_file_update_time_and_deterministic_id_tie_break(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JsonlContextStore(root)
            store.append("session-old", "user_input", {"text": "old"})
            store.append("session-a", "user_input", {"text": "new a"})
            store.append("session-b", "user_input", {"text": "new b"})
            os.utime(root / "session-old.jsonl", ns=(1_000_000_000, 1_000_000_000))
            os.utime(root / "session-a.jsonl", ns=(2_000_000_000, 2_000_000_000))
            os.utime(root / "session-b.jsonl", ns=(2_000_000_000, 2_000_000_000))
            (root / "ignored.txt").write_text("not a session", encoding="utf-8")
            discovery = JsonlSessionDiscovery(root)

            session_id = discovery.latest_session_id()

            self.assertEqual(session_id, "session-b")

    def test_latest_session_reports_error_when_no_sessions_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            JsonlContextStore(root)
            discovery = JsonlSessionDiscovery(root)

            with self.assertRaises(GearError) as raised:
                discovery.latest_session_id()

            self.assertEqual(raised.exception.error_type, "latest_session_not_found")
            self.assertTrue(raised.exception.recoverable)

    def test_resumed_session_appends_to_existing_jsonl_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JsonlContextStore(root)
            full_session_id = "12345678-1234-5678-1234-567812345678"
            store.append(full_session_id, "user_input", {"text": "before"})
            discovery = JsonlSessionDiscovery(root)

            session_id = discovery.resolve_session_id("12345678")
            store.append(session_id, "assistant_message", {"text": "after"})

            self.assertEqual(
                [path.name for path in root.glob("*.jsonl")],
                [f"{full_session_id}.jsonl"],
            )
            self.assertEqual(
                [event["payload"]["text"] for event in store.load(full_session_id)],
                ["before", "after"],
            )


if __name__ == "__main__":
    unittest.main()
