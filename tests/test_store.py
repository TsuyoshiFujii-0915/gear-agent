import json
import tempfile
import unittest
from pathlib import Path

from gear_agent.store.jsonl import JsonlContextStore


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


if __name__ == "__main__":
    unittest.main()
