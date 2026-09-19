from __future__ import annotations

import unittest

from gear_agent.artifact_privacy import ArtifactPrivacy
from gear_agent.config import load_config
from gear_agent.errors import GearError
from tests import test_headless as support


class ArtifactAmbiguousJsonTests(unittest.TestCase):
    setUp = support.HeadlessCliTests.setUp
    configure = support.HeadlessCliTests.configure

    def test_duplicate_keys_cannot_hide_secrets_from_projection(self) -> None:
        self.configure('http://localhost:1234/v1/responses', False, '')
        privacy = ArtifactPrivacy(load_config(self.config, {}), {})
        for arguments in ('{"Authorization":"hidden","Authorization":"[REDACTED]"}',
                          '{"nested":{"key":"first","key":"last"}}'):
            with self.subTest(arguments=arguments):
                with self.assertRaises(GearError) as raised:
                    privacy.serialize({'type': 'function_call', 'arguments': arguments})
                self.assertEqual(raised.exception.error_type, 'artifact_json_invalid')
                self.assertNotIn('hidden', str(raised.exception))

    def test_tool_result_data_is_not_interpreted_as_another_protocol_call(self) -> None:
        self.configure('http://localhost:1234/v1/responses', False, '')
        privacy = ArtifactPrivacy(load_config(self.config, {}), {})
        event = {'kind': 'tool_result', 'payload': {'result': {
            'type': 'function_call', 'arguments': 'ordinary returned data',
        }}}
        self.assertEqual(privacy.serialize(event), event)
