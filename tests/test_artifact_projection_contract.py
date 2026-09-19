from __future__ import annotations

import json
import unittest

from gear_agent.artifact_privacy import ArtifactPrivacy
from gear_agent.config import load_config
from tests import test_headless as support


class ArtifactProjectionContractTests(unittest.TestCase):
    setUp = support.HeadlessCliTests.setUp
    configure = support.HeadlessCliTests.configure

    def test_safe_protocol_arguments_keep_original_whitespace_and_key_order(self) -> None:
        self.configure('http://localhost:1234/v1/responses', False, '')
        privacy = ArtifactPrivacy(load_config(self.config, {}), {})
        item = {'type': 'function_call', 'arguments': ' { "z": 1, "a": [2] }\n'}
        self.assertEqual(privacy.serialize(item), item)

    def test_decoded_tool_argument_data_is_not_interpreted_as_protocol(self) -> None:
        self.configure('http://localhost:1234/v1/responses', False, '')
        privacy = ArtifactPrivacy(load_config(self.config, {}), {})
        arguments = {'type': 'function_call', 'arguments': 'arbitrary user data'}
        event = {'kind': 'tool_call', 'payload': {'name': 'test_tool', 'arguments': arguments}}
        self.assertEqual(privacy.serialize(event), event)
        model_item = {'type': 'function_call', 'arguments': json.dumps(arguments)}
        self.assertEqual(json.loads(privacy.serialize(model_item)['arguments']), arguments)
