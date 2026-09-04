import os
import tempfile
import unittest
from pathlib import Path

from gear_agent.config import (
    DEFAULT_CONFIG_TEXT,
    ReasoningReplayMode,
    discover_config_path,
    initialize_config,
    load_config,
)
from gear_agent.errors import GearError


class ConfigTests(unittest.TestCase):
    def test_loads_explicit_reasoning_replay_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "gear.toml"
            config_path.write_text(DEFAULT_CONFIG_TEXT, encoding="utf-8")

            disabled_config = load_config(config_path, {})

            config_path.write_text(
                DEFAULT_CONFIG_TEXT.replace(
                    'reasoning_replay = "none"',
                    'reasoning_replay = "encrypted"',
                ),
                encoding="utf-8",
            )
            encrypted_config = load_config(config_path, {})

            self.assertEqual(
                disabled_config.model.reasoning_replay,
                ReasoningReplayMode.NONE,
            )
            self.assertEqual(
                encrypted_config.model.reasoning_replay,
                ReasoningReplayMode.ENCRYPTED,
            )

    def test_rejects_invalid_reasoning_replay_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "gear.toml"
            config_path.write_text(
                DEFAULT_CONFIG_TEXT.replace(
                    'reasoning_replay = "none"',
                    'reasoning_replay = "automatic"',
                ),
                encoding="utf-8",
            )

            with self.assertRaises(GearError) as raised:
                load_config(config_path, {})

            self.assertEqual(raised.exception.error_type, "config_value_invalid")
            self.assertEqual(raised.exception.origin, "config")
            self.assertEqual(raised.exception.details["table"], "model")
            self.assertEqual(raised.exception.details["key"], "reasoning_replay")
            self.assertEqual(raised.exception.details["value"], "automatic")

    def test_loads_legacy_config_without_reasoning_replay_as_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "gear.toml"
            config_path.write_text(
                DEFAULT_CONFIG_TEXT.replace('reasoning_replay = "none"\n', ""),
                encoding="utf-8",
            )

            config = load_config(config_path, {})

            self.assertEqual(
                config.model.reasoning_replay,
                ReasoningReplayMode.NONE,
            )

    def test_allows_query_endpoint_for_encrypted_reasoning_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "gear.toml"
            config_path.write_text(
                DEFAULT_CONFIG_TEXT.replace(
                    'url = "http://localhost:1234/v1/responses"',
                    (
                        'url = "https://azure.example/openai/v1/responses'
                        '?api-version=v1"'
                    ),
                ).replace(
                    'reasoning_replay = "none"',
                    'reasoning_replay = "encrypted"',
                ),
                encoding="utf-8",
            )

            config = load_config(config_path, {})

            self.assertEqual(
                config.model.reasoning_replay,
                ReasoningReplayMode.ENCRYPTED,
            )
            self.assertEqual(
                config.model.url,
                "https://azure.example/openai/v1/responses?api-version=v1",
            )

    def test_loads_openai_api_key_from_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "gear.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[model]",
                        'url = "https://api.openai.com/v1/responses"',
                        'model = "gpt-5.5"',
                        'api_key_env = "OPENAI_API_KEY"',
                        'reasoning_replay = "none"',
                        "",
                        "[runtime]",
                        'workdir = "."',
                        'session_dir = ".gear/sessions"',
                        'network = "disabled"',
                        "max_iterations = 8",
                        "model_timeout_seconds = 120",
                        "",
                        "[tool]",
                        "shell_tool = true",
                        "file_read = true",
                        "file_write = true",
                        "apply_patch = true",
                        "glob = false",
                        "grep = false",
                        "web_search = false",
                        "web_fetch = false",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            environment = {"OPENAI_API_KEY": "secret-key"}

            config = load_config(config_path, environment)

            self.assertEqual(config.model.url, "https://api.openai.com/v1/responses")
            self.assertEqual(config.model.model, "gpt-5.5")
            self.assertEqual(config.model.api_key, "secret-key")
            self.assertEqual(config.runtime.workdir, Path("."))
            self.assertEqual(config.runtime.session_dir, Path(".gear/sessions"))
            self.assertFalse(config.runtime.network_enabled)
            self.assertEqual(config.runtime.max_iterations, 8)
            self.assertEqual(config.runtime.model_timeout_seconds, 120)
            self.assertTrue(config.tool.shell_tool)
            self.assertTrue(config.tool.file_read)
            self.assertTrue(config.tool.file_write)
            self.assertTrue(config.tool.apply_patch)
            self.assertFalse(config.tool.glob)
            self.assertFalse(config.tool.grep)
            self.assertFalse(config.tool.web_search)
            self.assertFalse(config.tool.web_fetch)
            self.assertIsNone(config.web_search)
            self.assertIsNone(config.web_fetch)

    def test_omits_authorization_when_api_key_env_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "gear.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[model]",
                        'url = "http://localhost:1234/v1/responses"',
                        'model = "local-model-id"',
                        'api_key_env = ""',
                        'reasoning_replay = "none"',
                        "",
                        "[runtime]",
                        'workdir = "."',
                        'session_dir = ".gear/sessions"',
                        'network = "disabled"',
                        "max_iterations = 8",
                        "model_timeout_seconds = 120",
                        "",
                        "[tool]",
                        "shell_tool = false",
                        "file_read = true",
                        "file_write = false",
                        "apply_patch = true",
                        "glob = false",
                        "grep = false",
                        "web_search = false",
                        "web_fetch = false",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path, os.environ)

            self.assertIsNone(config.model.api_key)
            self.assertFalse(config.tool.shell_tool)
            self.assertTrue(config.tool.file_read)
            self.assertFalse(config.tool.file_write)
            self.assertTrue(config.tool.apply_patch)
            self.assertFalse(config.tool.glob)
            self.assertFalse(config.tool.grep)
            self.assertFalse(config.tool.web_search)
            self.assertFalse(config.tool.web_fetch)
            self.assertIsNone(config.web_search)
            self.assertIsNone(config.web_fetch)

    def test_loads_enabled_filesystem_search_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "gear.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[model]",
                        'url = "http://localhost:1234/v1/responses"',
                        'model = "local-model-id"',
                        'api_key_env = ""',
                        'reasoning_replay = "none"',
                        "",
                        "[runtime]",
                        'workdir = "."',
                        'session_dir = ".gear/sessions"',
                        'network = "disabled"',
                        "max_iterations = 8",
                        "model_timeout_seconds = 120",
                        "",
                        "[tool]",
                        "shell_tool = false",
                        "file_read = false",
                        "file_write = false",
                        "apply_patch = false",
                        "glob = true",
                        "grep = true",
                        "web_search = false",
                        "web_fetch = false",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path, os.environ)

            self.assertTrue(config.tool.glob)
            self.assertTrue(config.tool.grep)
            self.assertFalse(config.tool.web_search)
            self.assertFalse(config.tool.web_fetch)

    def test_loads_enabled_tavily_web_search_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "gear.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[model]",
                        'url = "http://localhost:1234/v1/responses"',
                        'model = "local-model-id"',
                        'api_key_env = ""',
                        'reasoning_replay = "none"',
                        "",
                        "[runtime]",
                        'workdir = "."',
                        'session_dir = ".gear/sessions"',
                        'network = "disabled"',
                        "max_iterations = 8",
                        "model_timeout_seconds = 120",
                        "",
                        "[tool]",
                        "shell_tool = false",
                        "file_read = true",
                        "file_write = false",
                        "apply_patch = true",
                        "glob = false",
                        "grep = false",
                        "web_search = true",
                        "web_fetch = false",
                        "",
                        "[web_search]",
                        'api_key_env = "TAVILY_API_KEY"',
                        'search_depth = "basic"',
                        "max_results = 5",
                        "timeout_seconds = 20",
                        "include_answer = true",
                        "include_raw_content = false",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path, {"TAVILY_API_KEY": "tvly-secret"})

            self.assertTrue(config.tool.web_search)
            self.assertIsNotNone(config.web_search)
            if config.web_search is None:
                raise AssertionError("web_search config should be present.")
            self.assertEqual(config.web_search.api_key, "tvly-secret")
            self.assertEqual(config.web_search.search_depth, "basic")
            self.assertEqual(config.web_search.max_results, 5)
            self.assertEqual(config.web_search.timeout_seconds, 20)
            self.assertTrue(config.web_search.include_answer)
            self.assertFalse(config.web_search.include_raw_content)
            self.assertIsNone(config.web_fetch)

    def test_loads_enabled_tavily_web_fetch_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "gear.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[model]",
                        'url = "http://localhost:1234/v1/responses"',
                        'model = "local-model-id"',
                        'api_key_env = ""',
                        'reasoning_replay = "none"',
                        "",
                        "[runtime]",
                        'workdir = "."',
                        'session_dir = ".gear/sessions"',
                        'network = "disabled"',
                        "max_iterations = 8",
                        "model_timeout_seconds = 120",
                        "",
                        "[tool]",
                        "shell_tool = false",
                        "file_read = true",
                        "file_write = false",
                        "apply_patch = true",
                        "glob = false",
                        "grep = false",
                        "web_search = false",
                        "web_fetch = true",
                        "",
                        "[web_fetch]",
                        'api_key_env = "TAVILY_API_KEY"',
                        'extract_depth = "basic"',
                        'content_format = "markdown"',
                        "timeout_seconds = 20",
                        "include_images = false",
                        "include_favicon = true",
                        "max_content_chars = 20000",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path, {"TAVILY_API_KEY": "tvly-secret"})

            self.assertFalse(config.tool.web_search)
            self.assertTrue(config.tool.web_fetch)
            self.assertIsNone(config.web_search)
            self.assertIsNotNone(config.web_fetch)
            if config.web_fetch is None:
                raise AssertionError("web_fetch config should be present.")
            self.assertEqual(config.web_fetch.api_key, "tvly-secret")
            self.assertEqual(config.web_fetch.extract_depth, "basic")
            self.assertEqual(config.web_fetch.content_format, "markdown")
            self.assertEqual(config.web_fetch.timeout_seconds, 20)
            self.assertFalse(config.web_fetch.include_images)
            self.assertTrue(config.web_fetch.include_favicon)
            self.assertEqual(config.web_fetch.max_content_chars, 20_000)

    def test_fails_when_web_search_enabled_without_config_table(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "gear.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[model]",
                        'url = "http://localhost:1234/v1/responses"',
                        'model = "local-model-id"',
                        'api_key_env = ""',
                        'reasoning_replay = "none"',
                        "",
                        "[runtime]",
                        'workdir = "."',
                        'session_dir = ".gear/sessions"',
                        'network = "disabled"',
                        "max_iterations = 8",
                        "model_timeout_seconds = 120",
                        "",
                        "[tool]",
                        "shell_tool = false",
                        "file_read = true",
                        "file_write = false",
                        "apply_patch = true",
                        "glob = false",
                        "grep = false",
                        "web_search = true",
                        "web_fetch = false",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(GearError) as error:
                load_config(config_path, {})

            self.assertEqual(error.exception.origin, "config")
            self.assertIn("[web_search]", error.exception.message)

    def test_fails_when_web_search_api_key_environment_variable_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "gear.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[model]",
                        'url = "http://localhost:1234/v1/responses"',
                        'model = "local-model-id"',
                        'api_key_env = ""',
                        'reasoning_replay = "none"',
                        "",
                        "[runtime]",
                        'workdir = "."',
                        'session_dir = ".gear/sessions"',
                        'network = "disabled"',
                        "max_iterations = 8",
                        "model_timeout_seconds = 120",
                        "",
                        "[tool]",
                        "shell_tool = false",
                        "file_read = true",
                        "file_write = false",
                        "apply_patch = true",
                        "glob = false",
                        "grep = false",
                        "web_search = true",
                        "web_fetch = false",
                        "",
                        "[web_search]",
                        'api_key_env = "TAVILY_API_KEY"',
                        'search_depth = "basic"',
                        "max_results = 5",
                        "timeout_seconds = 20",
                        "include_answer = false",
                        "include_raw_content = false",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(GearError) as error:
                load_config(config_path, {})

            self.assertEqual(error.exception.origin, "config")
            self.assertIn("TAVILY_API_KEY", error.exception.message)

    def test_fails_when_web_search_depth_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "gear.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[model]",
                        'url = "http://localhost:1234/v1/responses"',
                        'model = "local-model-id"',
                        'api_key_env = ""',
                        'reasoning_replay = "none"',
                        "",
                        "[runtime]",
                        'workdir = "."',
                        'session_dir = ".gear/sessions"',
                        'network = "disabled"',
                        "max_iterations = 8",
                        "model_timeout_seconds = 120",
                        "",
                        "[tool]",
                        "shell_tool = false",
                        "file_read = true",
                        "file_write = false",
                        "apply_patch = true",
                        "glob = false",
                        "grep = false",
                        "web_search = true",
                        "web_fetch = false",
                        "",
                        "[web_search]",
                        'api_key_env = "TAVILY_API_KEY"',
                        'search_depth = "slow"',
                        "max_results = 5",
                        "timeout_seconds = 20",
                        "include_answer = false",
                        "include_raw_content = false",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(GearError) as error:
                load_config(config_path, {"TAVILY_API_KEY": "tvly-secret"})

            self.assertEqual(error.exception.origin, "config")
            self.assertIn("web_search.search_depth", error.exception.message)

    def test_fails_when_web_fetch_enabled_without_config_table(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "gear.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[model]",
                        'url = "http://localhost:1234/v1/responses"',
                        'model = "local-model-id"',
                        'api_key_env = ""',
                        'reasoning_replay = "none"',
                        "",
                        "[runtime]",
                        'workdir = "."',
                        'session_dir = ".gear/sessions"',
                        'network = "disabled"',
                        "max_iterations = 8",
                        "model_timeout_seconds = 120",
                        "",
                        "[tool]",
                        "shell_tool = false",
                        "file_read = true",
                        "file_write = false",
                        "apply_patch = true",
                        "glob = false",
                        "grep = false",
                        "web_search = false",
                        "web_fetch = true",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(GearError) as error:
                load_config(config_path, {})

            self.assertEqual(error.exception.origin, "config")
            self.assertIn("[web_fetch]", error.exception.message)

    def test_fails_when_web_fetch_depth_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "gear.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[model]",
                        'url = "http://localhost:1234/v1/responses"',
                        'model = "local-model-id"',
                        'api_key_env = ""',
                        'reasoning_replay = "none"',
                        "",
                        "[runtime]",
                        'workdir = "."',
                        'session_dir = ".gear/sessions"',
                        'network = "disabled"',
                        "max_iterations = 8",
                        "model_timeout_seconds = 120",
                        "",
                        "[tool]",
                        "shell_tool = false",
                        "file_read = true",
                        "file_write = false",
                        "apply_patch = true",
                        "glob = false",
                        "grep = false",
                        "web_search = false",
                        "web_fetch = true",
                        "",
                        "[web_fetch]",
                        'api_key_env = "TAVILY_API_KEY"',
                        'extract_depth = "deep"',
                        'content_format = "markdown"',
                        "timeout_seconds = 20",
                        "include_images = false",
                        "include_favicon = true",
                        "max_content_chars = 20000",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(GearError) as error:
                load_config(config_path, {"TAVILY_API_KEY": "tvly-secret"})

            self.assertEqual(error.exception.origin, "config")
            self.assertIn("web_fetch.extract_depth", error.exception.message)

    def test_fails_when_web_fetch_content_format_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "gear.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[model]",
                        'url = "http://localhost:1234/v1/responses"',
                        'model = "local-model-id"',
                        'api_key_env = ""',
                        'reasoning_replay = "none"',
                        "",
                        "[runtime]",
                        'workdir = "."',
                        'session_dir = ".gear/sessions"',
                        'network = "disabled"',
                        "max_iterations = 8",
                        "model_timeout_seconds = 120",
                        "",
                        "[tool]",
                        "shell_tool = false",
                        "file_read = true",
                        "file_write = false",
                        "apply_patch = true",
                        "glob = false",
                        "grep = false",
                        "web_search = false",
                        "web_fetch = true",
                        "",
                        "[web_fetch]",
                        'api_key_env = "TAVILY_API_KEY"',
                        'extract_depth = "basic"',
                        'content_format = "html"',
                        "timeout_seconds = 20",
                        "include_images = false",
                        "include_favicon = true",
                        "max_content_chars = 20000",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(GearError) as error:
                load_config(config_path, {"TAVILY_API_KEY": "tvly-secret"})

            self.assertEqual(error.exception.origin, "config")
            self.assertIn("web_fetch.content_format", error.exception.message)

    def test_fails_when_named_api_key_environment_variable_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "gear.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[model]",
                        'url = "https://api.openai.com/v1/responses"',
                        'model = "gpt-5.5"',
                        'api_key_env = "OPENAI_API_KEY"',
                        'reasoning_replay = "none"',
                        "",
                        "[runtime]",
                        'workdir = "."',
                        'session_dir = ".gear/sessions"',
                        'network = "disabled"',
                        "max_iterations = 8",
                        "model_timeout_seconds = 120",
                        "",
                        "[tool]",
                        "shell_tool = true",
                        "file_read = true",
                        "file_write = true",
                        "apply_patch = true",
                        "glob = false",
                        "grep = false",
                        "web_search = false",
                        "web_fetch = false",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(GearError) as error:
                load_config(config_path, {})

            self.assertEqual(error.exception.origin, "config")
            self.assertIn("OPENAI_API_KEY", error.exception.message)

    def test_fails_when_network_value_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "gear.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[model]",
                        'url = "http://localhost:1234/v1/responses"',
                        'model = "local-model-id"',
                        'api_key_env = ""',
                        'reasoning_replay = "none"',
                        "",
                        "[runtime]",
                        'workdir = "."',
                        'session_dir = ".gear/sessions"',
                        'network = "maybe"',
                        "max_iterations = 8",
                        "model_timeout_seconds = 120",
                        "",
                        "[tool]",
                        "shell_tool = true",
                        "file_read = true",
                        "file_write = true",
                        "apply_patch = true",
                        "glob = false",
                        "grep = false",
                        "web_search = false",
                        "web_fetch = false",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(GearError) as error:
                load_config(config_path, os.environ)

            self.assertEqual(error.exception.origin, "config")
            self.assertIn("runtime.network", error.exception.message)

    def test_fails_when_tool_table_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "gear.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[model]",
                        'url = "http://localhost:1234/v1/responses"',
                        'model = "local-model-id"',
                        'api_key_env = ""',
                        'reasoning_replay = "none"',
                        "",
                        "[runtime]",
                        'workdir = "."',
                        'session_dir = ".gear/sessions"',
                        'network = "disabled"',
                        "max_iterations = 8",
                        "model_timeout_seconds = 120",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(GearError) as error:
                load_config(config_path, os.environ)

            self.assertEqual(error.exception.origin, "config")
            self.assertIn("[tool]", error.exception.message)

    def test_fails_when_tool_value_is_not_boolean(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "gear.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[model]",
                        'url = "http://localhost:1234/v1/responses"',
                        'model = "local-model-id"',
                        'api_key_env = ""',
                        'reasoning_replay = "none"',
                        "",
                        "[runtime]",
                        'workdir = "."',
                        'session_dir = ".gear/sessions"',
                        'network = "disabled"',
                        "max_iterations = 8",
                        "model_timeout_seconds = 120",
                        "",
                        "[tool]",
                        'shell_tool = "enabled"',
                        "file_read = true",
                        "file_write = true",
                        "apply_patch = true",
                        "glob = false",
                        "grep = false",
                        "web_search = false",
                        "web_fetch = false",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(GearError) as error:
                load_config(config_path, os.environ)

            self.assertEqual(error.exception.origin, "config")
            self.assertIn("tool.shell_tool", error.exception.message)

    def test_fails_when_tool_key_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "gear.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[model]",
                        'url = "http://localhost:1234/v1/responses"',
                        'model = "local-model-id"',
                        'api_key_env = ""',
                        'reasoning_replay = "none"',
                        "",
                        "[runtime]",
                        'workdir = "."',
                        'session_dir = ".gear/sessions"',
                        'network = "disabled"',
                        "max_iterations = 8",
                        "model_timeout_seconds = 120",
                        "",
                        "[tool]",
                        "shell_tool = true",
                        "file_read = true",
                        "file_write = true",
                        "apply_patch = true",
                        "glob = false",
                        "grep = false",
                        "web_search = false",
                        "web_fetch = false",
                        "browser = true",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(GearError) as error:
                load_config(config_path, os.environ)

            self.assertEqual(error.exception.origin, "config")
            self.assertIn("tool.browser", error.exception.message)

    def test_discovers_project_config_before_user_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project_config = root / ".gear" / "config.toml"
            user_config = root / "home" / ".gear" / "config.toml"
            project_config.parent.mkdir(parents=True)
            user_config.parent.mkdir(parents=True)
            project_config.write_text("project", encoding="utf-8")
            user_config.write_text("user", encoding="utf-8")

            discovered = discover_config_path(root, root / "home")

            self.assertEqual(discovered, project_config)

    def test_discovers_parent_project_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "src" / "package"
            nested.mkdir(parents=True)
            project_config = root / ".gear" / "config.toml"
            project_config.parent.mkdir()
            project_config.write_text("project", encoding="utf-8")

            discovered = discover_config_path(nested, root / "home")

            self.assertEqual(discovered, project_config)

    def test_discovers_user_config_when_project_config_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            user_config = root / "home" / ".gear" / "config.toml"
            user_config.parent.mkdir(parents=True)
            user_config.write_text("user", encoding="utf-8")

            discovered = discover_config_path(root, root / "home")

            self.assertEqual(discovered, user_config)

    def test_initializes_project_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            path = initialize_config("project", root, root / "home")

            self.assertEqual(path, root / ".gear" / "config.toml")
            config_text = path.read_text(encoding="utf-8")
            self.assertIn("[model]", config_text)
            self.assertIn('reasoning_replay = "none"', config_text)
            self.assertIn("[tool]", config_text)
            self.assertIn("[web_search]", config_text)
            self.assertIn("[web_fetch]", config_text)
            self.assertIn("glob = true", config_text)
            self.assertIn("grep = true", config_text)
            self.assertIn('api_key_env = "TAVILY_API_KEY"', config_text)
            self.assertIn('search_depth = "basic"', config_text)
            self.assertIn('extract_depth = "basic"', config_text)
            self.assertIn("max_results = 5", config_text)
            self.assertLess(config_text.index("[tool]"), config_text.index("[runtime]"))
            self.assertLess(config_text.index("[web_search]"), config_text.index("[runtime]"))
            self.assertLess(config_text.index("[web_fetch]"), config_text.index("[runtime]"))
            config = load_config(path, {})
            self.assertTrue(config.tool.glob)
            self.assertTrue(config.tool.grep)
            self.assertFalse(config.tool.web_search)
            self.assertFalse(config.tool.web_fetch)
            self.assertIsNone(config.web_search)
            self.assertIsNone(config.web_fetch)

    def test_initializes_user_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            path = initialize_config("user", root, root / "home")

            self.assertEqual(path, root / "home" / ".gear" / "config.toml")
            config_text = path.read_text(encoding="utf-8")
            self.assertIn("[runtime]", config_text)
            self.assertIn("shell_tool = true", config_text)
            self.assertIn("glob = true", config_text)
            self.assertIn("grep = true", config_text)
            self.assertIn("[web_search]", config_text)
            self.assertIn("[web_fetch]", config_text)
            self.assertIn("include_raw_content = false", config_text)
            self.assertIn("include_favicon = true", config_text)
            self.assertLess(config_text.index("[tool]"), config_text.index("[runtime]"))

    def test_initialize_fails_when_config_already_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / ".gear" / "config.toml"
            path.parent.mkdir()
            path.write_text("existing", encoding="utf-8")

            with self.assertRaises(GearError) as error:
                initialize_config("project", root, root / "home")

            self.assertEqual(error.exception.origin, "config")


if __name__ == "__main__":
    unittest.main()
