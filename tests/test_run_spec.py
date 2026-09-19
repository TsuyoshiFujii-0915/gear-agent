from dataclasses import asdict, replace
from pathlib import Path
import json
import tempfile
import unittest

from gear_agent.agent.events import SilentAgentLoopEventSink
from gear_agent.config import DEFAULT_CONFIG_TEXT, load_config
from gear_agent.headless import TaskPrompt, describe_run
from gear_agent.runtime import build_agent_runtime


class RunSpecTests(unittest.TestCase):
    def test_effective_run_inputs_are_discoverable_without_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'config.toml'
            path.write_text(DEFAULT_CONFIG_TEXT)
            config = load_config(path, {})
            config = replace(config, model=replace(
                config.model, api_key='model-secret',
                url='https://user:password@example.com/v1/responses?token=model-secret',
            ))
            runtime = replace(config.runtime, workdir=root, session_dir=root / 'sessions',
                              max_iterations=4, network_enabled=True)
            agent = build_agent_runtime(config, runtime, SilentAgentLoopEventSink())
            spec = describe_run(agent, 'session-123', TaskPrompt('task', 'inline'))
            self.assertEqual(spec.session_id, 'session-123')
            self.assertEqual(spec.workspace, str(root.resolve()))
            self.assertEqual(spec.prompt_source, 'inline')
            self.assertEqual(spec.runtime['max_iterations'], 4)
            self.assertTrue(spec.runtime['network_enabled'])
            self.assertTrue(spec.tools['file_read'])
            self.assertTrue(spec.capabilities['streaming'])
            self.assertEqual(spec.adapter_kind, 'ResponsesModelAdapter')
            self.assertFalse(spec.context_budget['auto_compaction'])
            serialized = json.dumps(asdict(spec))
            for secret in ('model-secret', 'password', 'api_key', 'https://'):
                self.assertNotIn(secret, serialized)
            rotated = replace(config, model=replace(config.model, api_key='rotated-secret',
                url='https://user:rotated-password@example.com/v1/responses?token=rotated-secret'))
            other = build_agent_runtime(rotated, runtime, SilentAgentLoopEventSink())
            self.assertEqual(spec.model, describe_run(other, 'other', TaskPrompt('task', 'inline')).model)
