from __future__ import annotations

import json
import unittest

from tests import test_headless as support


class ArtifactUsageTests(unittest.TestCase):
    setUp = support.HeadlessCliTests.setUp
    configure = support.HeadlessCliTests.configure
    invoke = support.HeadlessCliTests.invoke
    history = support.HeadlessCliTests.history

    def test_invalid_usage_does_not_discard_canonical_received_response(self) -> None:
        response = support.message('received')
        response['usage'] = {'input_tokens': -1}
        with support.model_endpoint([support.json_response(response)]) as (url, requests):
            self.configure(url, False, '')
            result = self.invoke(['run', '--prompt', 'task', '--run-dir', str(self.root / 'run')])
        self.assertEqual(result.returncode, 3)
        received = [e for e in self.history(result) if e['kind'] == 'model_response']
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]['payload'], response)
        events = [json.loads(line) for line in (self.root / 'run/events.jsonl').read_text().splitlines()]
        self.assertIn(received[0], events)
