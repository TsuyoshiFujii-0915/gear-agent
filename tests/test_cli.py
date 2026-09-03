import unittest
from pathlib import Path

from gear_agent.cli import _build_parser


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


if __name__ == "__main__":
    unittest.main()
