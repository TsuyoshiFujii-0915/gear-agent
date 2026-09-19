from __future__ import annotations

import unittest

from gear_agent.artifact_privacy import ArtifactPrivacy
from gear_agent.config import load_config
from tests import test_headless as support


class ArtifactPrivacyTests(unittest.TestCase):
    setUp = support.HeadlessCliTests.setUp
    configure = support.HeadlessCliTests.configure

    def test_common_credential_fields_are_removed_without_losing_token_metrics(self) -> None:
        self.configure('http://localhost:1234/v1/responses', False, '')
        privacy = ArtifactPrivacy(load_config(self.config, {}), {})
        value = privacy.serialize({'access_token': 'access-value', 'refresh_token': 'refresh-value',
                                   'client_secret': 'client-value', 'X-API-Key': 'api-value',
                                   'Authorization': 'Bearer auth-value', 'total_tokens': 20,
                                   'input_tokens': 11, 'output_tokens': None})
        for name in ('access_token', 'refresh_token', 'client_secret', 'X-API-Key', 'Authorization'):
            self.assertEqual(value[name], '[REDACTED]')
        self.assertEqual(value['total_tokens'], 20)
        self.assertEqual(value['input_tokens'], 11)
        self.assertIsNone(value['output_tokens'])

    def test_embedded_json_authorization_and_credentials_are_redacted(self) -> None:
        self.configure('http://localhost:1234/v1/responses', False, '')
        privacy = ArtifactPrivacy(load_config(self.config, {}), {})
        text = '{"Authorization": "Bearer header-private", "api_key": "query-private"}'
        redacted = privacy.text(text)
        self.assertNotIn('header-private', redacted)
        self.assertNotIn('query-private', redacted)

    def test_known_secrets_are_removed_when_url_encoded(self) -> None:
        self.configure('http://localhost:1234/v1/responses', False, '')
        privacy = ArtifactPrivacy(load_config(self.config, {}), {'PRIVATE_SECRET': 'secret /?value'})
        for text in ('secret /?value', 'secret%20%2F%3Fvalue', 'secret+%2F%3Fvalue'):
            self.assertEqual(privacy.text(text), '[REDACTED]')
