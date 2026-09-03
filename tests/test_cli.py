import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from gear_agent.cli import _build_parser, _runtime_from_args, _session_id_from_args
from gear_agent.config import RuntimeConfig
from gear_agent.store.jsonl import JsonlContextStore


class CliTests(unittest.TestCase):
    def test_help_uses_gear_agent_branding(self) -> None:
        help_text = _build_parser().format_help()

        self.assertIn("Gear Agent", help_text)
        self.assertNotIn("Gear Code", help_text)

    def test_discovers_config_by_default(self) -> None:
        args = _build_parser().parse_args([])

        self.assertIsNone(args.config)
        self.assertIsNone(args.command)

    def test_accepts_runtime_overrides(self) -> None:
        args = _build_parser().parse_args(
            [
                "--config",
                "custom.toml",
                "--workdir",
                ".",
                "--session-dir",
                ".gear/sessions",
                "--network",
                "enabled",
                "--max-iterations",
                "4",
                "--model-timeout-seconds",
                "30",
            ]
        )

        self.assertEqual(args.config, Path("custom.toml"))
        self.assertEqual(args.workdir, Path("."))
        self.assertEqual(args.session_dir, Path(".gear/sessions"))
        self.assertEqual(args.network, "enabled")
        self.assertEqual(args.max_iterations, 4)
        self.assertEqual(args.model_timeout_seconds, 30)

    def test_accepts_project_init_command(self) -> None:
        args = _build_parser().parse_args(["init"])

        self.assertEqual(args.command, "init")
        self.assertEqual(args.scope, "project")

    def test_accepts_user_init_command(self) -> None:
        args = _build_parser().parse_args(["init", "--scope", "user"])

        self.assertEqual(args.command, "init")
        self.assertEqual(args.scope, "user")

    def test_accepts_resume_with_session_id_prefix(self) -> None:
        args = _build_parser().parse_args(["resume", "12345678"])

        self.assertEqual(args.command, "resume")
        self.assertEqual(args.session_reference, "12345678")
        self.assertFalse(args.latest)

    def test_accepts_resume_latest(self) -> None:
        args = _build_parser().parse_args(["resume", "--latest"])

        self.assertEqual(args.command, "resume")
        self.assertIsNone(args.session_reference)
        self.assertTrue(args.latest)

    def test_resume_rejects_session_reference_with_latest(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                _build_parser().parse_args(["resume", "12345678", "--latest"])

    def test_resume_requires_session_reference_or_latest(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                _build_parser().parse_args(["resume"])

    def test_default_command_generates_fresh_session_id(self) -> None:
        args = _build_parser().parse_args([])
        expected = UUID("12345678-1234-5678-1234-567812345678")

        with patch("gear_agent.cli.uuid4", return_value=expected):
            session_id = _session_id_from_args(args, Path("unused"))

        self.assertEqual(session_id, str(expected))

    def test_resume_uses_cli_overridden_session_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            configured_session_dir = root / "configured-sessions"
            overridden_session_dir = root / "overridden-sessions"
            JsonlContextStore(configured_session_dir).append(
                "configured-session",
                "user_input",
                {"text": "configured"},
            )
            JsonlContextStore(overridden_session_dir).append(
                "overridden-session",
                "user_input",
                {"text": "overridden"},
            )
            configured_runtime = RuntimeConfig(
                workdir=root,
                session_dir=configured_session_dir,
                network_enabled=False,
                max_iterations=8,
                model_timeout_seconds=120,
            )
            args = _build_parser().parse_args(
                ["--session-dir", str(overridden_session_dir), "resume", "--latest"]
            )

            runtime = _runtime_from_args(configured_runtime, args)
            session_id = _session_id_from_args(args, runtime.session_dir)

            self.assertEqual(session_id, "overridden-session")


if __name__ == "__main__":
    unittest.main()
