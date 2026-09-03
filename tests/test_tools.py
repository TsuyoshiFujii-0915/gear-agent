import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gear_agent.config import ToolConfig, WebFetchConfig, WebSearchConfig
from gear_agent.errors import GearError
from gear_agent.tools.configured import build_configured_tools
from gear_agent.tools.filesystem import FileReadTool, FileWriteTool
from gear_agent.tools.filesystem_search import GlobTool, GrepTool
from gear_agent.tools.registry import ToolRegistry
from gear_agent.tools.runtimes import DockerShellRuntime, ShellRuntime
from gear_agent.tools.shell import ShellTool


class FakeShellRuntime(ShellRuntime):
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path, int]] = []

    def run(self, command: str, workdir: Path, timeout_seconds: int) -> dict[str, object]:
        self.calls.append((command, workdir, timeout_seconds))
        return {
            "exit_code": 0,
            "stdout": "ok",
            "stderr": "",
            "timed_out": False,
        }


class ToolTests(unittest.TestCase):
    def test_unknown_tool_name_fails(self) -> None:
        registry = ToolRegistry([])

        with self.assertRaises(GearError) as error:
            registry.run("missing_tool", {})

        self.assertEqual(error.exception.origin, "tool_registry")

    def test_file_read_rejects_path_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            workspace.mkdir()
            outside_file = Path(temp_dir) / "secret.txt"
            outside_file.write_text("secret", encoding="utf-8")
            tool = FileReadTool(workspace)

            with self.assertRaises(GearError) as error:
                tool.run({"path": "../secret.txt"})

            self.assertEqual(error.exception.origin, "file_read")

    def test_file_write_requires_existing_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            tool = FileWriteTool(workspace)

            with self.assertRaises(GearError) as error:
                tool.run({"path": "missing/file.txt", "content": "hello"})

            self.assertEqual(error.exception.origin, "file_write")

    def test_glob_returns_workspace_relative_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "src").mkdir()
            (workspace / "src" / "main.py").write_text("print('ok')\n", encoding="utf-8")
            (workspace / "README.md").write_text("# Test\n", encoding="utf-8")
            tool = GlobTool(workspace)

            result = tool.run({"pattern": "**/*.py", "max_results": 10})

            self.assertEqual(
                result,
                {
                    "pattern": "**/*.py",
                    "matches": [{"path": "src/main.py", "type": "file"}],
                    "truncated": False,
                },
            )

    def test_glob_truncates_after_max_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "a.txt").write_text("a\n", encoding="utf-8")
            (workspace / "b.txt").write_text("b\n", encoding="utf-8")
            tool = GlobTool(workspace)

            result = tool.run({"pattern": "*.txt", "max_results": 1})

            self.assertEqual(result["matches"], [{"path": "a.txt", "type": "file"}])
            self.assertIs(result["truncated"], True)

    def test_glob_rejects_path_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            tool = GlobTool(workspace)

            with self.assertRaises(GearError) as error:
                tool.run({"pattern": "../*.txt", "max_results": 10})

            self.assertEqual(error.exception.origin, "glob")

    def test_grep_returns_workspace_relative_line_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "src").mkdir()
            (workspace / "src" / "main.py").write_text(
                "def main():\n    return 'needle'\n",
                encoding="utf-8",
            )
            (workspace / "README.md").write_text("needle\n", encoding="utf-8")
            tool = GrepTool(workspace)

            result = tool.run({"path": "src", "pattern": "needle", "max_results": 10})

            self.assertEqual(
                result,
                {
                    "path": "src",
                    "pattern": "needle",
                    "matches": [
                        {
                            "path": "src/main.py",
                            "line": 2,
                            "text": "    return 'needle'",
                        }
                    ],
                    "truncated": False,
                },
            )

    def test_grep_truncates_after_max_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "one.txt").write_text("needle\n", encoding="utf-8")
            (workspace / "two.txt").write_text("needle\n", encoding="utf-8")
            tool = GrepTool(workspace)

            result = tool.run({"path": ".", "pattern": "needle", "max_results": 1})

            self.assertEqual(
                result["matches"],
                [{"path": "one.txt", "line": 1, "text": "needle"}],
            )
            self.assertIs(result["truncated"], True)

    def test_grep_rejects_invalid_regex(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = GrepTool(Path(temp_dir))

            with self.assertRaises(GearError) as error:
                tool.run({"path": ".", "pattern": "[", "max_results": 10})

            self.assertEqual(error.exception.origin, "grep")
            self.assertIn("regex", error.exception.message)

    def test_grep_rejects_non_utf8_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "binary.dat").write_bytes(b"\xff")
            tool = GrepTool(workspace)

            with self.assertRaises(GearError) as error:
                tool.run({"path": "binary.dat", "pattern": "needle", "max_results": 10})

            self.assertEqual(error.exception.origin, "grep")

    def test_shell_requires_timeout_seconds(self) -> None:
        runtime = FakeShellRuntime()
        tool = ShellTool(Path.cwd(), runtime)

        with self.assertRaises(GearError) as error:
            tool.run({"command": "echo ok", "workdir": "."})

        self.assertEqual(error.exception.origin, "shell")

    def test_docker_shell_timeout_returns_json_serializable_text_output(self) -> None:
        runtime = DockerShellRuntime(Path.cwd(), "python:3.11-slim", False)
        timeout = subprocess.TimeoutExpired(
            cmd=["docker"],
            timeout=1,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

        with patch("subprocess.run", side_effect=timeout):
            result = runtime.run("sleep 2", Path.cwd(), 1)

        self.assertEqual(result["stdout"], "partial stdout")
        self.assertEqual(result["stderr"], "partial stderr")
        self.assertIs(result["timed_out"], True)
        json.dumps(result)

    def test_tool_schemas_explain_workspace_relative_paths(self) -> None:
        runtime = FakeShellRuntime()
        workspace = Path.cwd()
        shell_schema = ShellTool(workspace, runtime).schema()
        read_schema = FileReadTool(workspace).schema()
        write_schema = FileWriteTool(workspace).schema()

        shell_parameters = shell_schema["parameters"]
        self.assertIsInstance(shell_parameters, dict)
        shell_properties = shell_parameters["properties"]
        self.assertIsInstance(shell_properties, dict)
        shell_workdir = shell_properties["workdir"]
        self.assertIsInstance(shell_workdir, dict)
        self.assertIn("relative", str(shell_workdir["description"]))
        self.assertIn("'.'", str(shell_workdir["description"]))
        self.assertIn("Absolute paths are rejected", str(shell_workdir["description"]))

        read_parameters = read_schema["parameters"]
        write_parameters = write_schema["parameters"]
        self.assertIsInstance(read_parameters, dict)
        self.assertIsInstance(write_parameters, dict)
        read_properties = read_parameters["properties"]
        write_properties = write_parameters["properties"]
        self.assertIsInstance(read_properties, dict)
        self.assertIsInstance(write_properties, dict)
        self.assertIn("relative", str(read_properties["path"]["description"]))
        self.assertIn("relative", str(write_properties["path"]["description"]))

    def test_build_configured_tools_exposes_only_enabled_tool_schemas(self) -> None:
        runtime = FakeShellRuntime()
        workspace = Path.cwd()
        tool_config = ToolConfig(
            shell_tool=False,
            file_read=True,
            file_write=False,
            apply_patch=True,
            glob=False,
            grep=False,
            web_search=False,
            web_fetch=False,
        )

        registry = ToolRegistry(
            build_configured_tools(tool_config, None, None, workspace, runtime)
        )

        schema_names = [schema["name"] for schema in registry.schemas()]
        self.assertEqual(schema_names, ["file_read", "apply_patch"])

    def test_build_configured_tools_exposes_web_search_when_enabled(self) -> None:
        runtime = FakeShellRuntime()
        workspace = Path.cwd()
        tool_config = ToolConfig(
            shell_tool=False,
            file_read=False,
            file_write=False,
            apply_patch=False,
            glob=False,
            grep=False,
            web_search=True,
            web_fetch=False,
        )
        web_search_config = WebSearchConfig(
            api_key="tvly-secret",
            search_depth="basic",
            max_results=5,
            timeout_seconds=20,
            include_answer=False,
            include_raw_content=False,
        )

        registry = ToolRegistry(
            build_configured_tools(tool_config, web_search_config, None, workspace, runtime)
        )

        schema_names = [schema["name"] for schema in registry.schemas()]
        self.assertEqual(schema_names, ["web_search"])

    def test_build_configured_tools_exposes_web_fetch_when_enabled(self) -> None:
        runtime = FakeShellRuntime()
        workspace = Path.cwd()
        tool_config = ToolConfig(
            shell_tool=False,
            file_read=False,
            file_write=False,
            apply_patch=False,
            glob=False,
            grep=False,
            web_search=False,
            web_fetch=True,
        )
        web_fetch_config = WebFetchConfig(
            api_key="tvly-secret",
            extract_depth="basic",
            content_format="markdown",
            timeout_seconds=20,
            include_images=False,
            include_favicon=True,
            max_content_chars=20_000,
        )

        registry = ToolRegistry(
            build_configured_tools(tool_config, None, web_fetch_config, workspace, runtime)
        )

        schema_names = [schema["name"] for schema in registry.schemas()]
        self.assertEqual(schema_names, ["web_fetch"])

    def test_build_configured_tools_requires_web_search_config_when_enabled(self) -> None:
        runtime = FakeShellRuntime()
        workspace = Path.cwd()
        tool_config = ToolConfig(
            shell_tool=False,
            file_read=False,
            file_write=False,
            apply_patch=False,
            glob=False,
            grep=False,
            web_search=True,
            web_fetch=False,
        )

        with self.assertRaises(GearError) as error:
            build_configured_tools(tool_config, None, None, workspace, runtime)

        self.assertEqual(error.exception.origin, "tool_config")
        self.assertIn("web_search", error.exception.message)

    def test_build_configured_tools_requires_web_fetch_config_when_enabled(self) -> None:
        runtime = FakeShellRuntime()
        workspace = Path.cwd()
        tool_config = ToolConfig(
            shell_tool=False,
            file_read=False,
            file_write=False,
            apply_patch=False,
            glob=False,
            grep=False,
            web_search=False,
            web_fetch=True,
        )

        with self.assertRaises(GearError) as error:
            build_configured_tools(tool_config, None, None, workspace, runtime)

        self.assertEqual(error.exception.origin, "tool_config")
        self.assertIn("web_fetch", error.exception.message)

    def test_build_configured_tools_exposes_filesystem_search_tools_when_enabled(self) -> None:
        runtime = FakeShellRuntime()
        workspace = Path.cwd()
        tool_config = ToolConfig(
            shell_tool=False,
            file_read=False,
            file_write=False,
            apply_patch=False,
            glob=True,
            grep=True,
            web_search=False,
            web_fetch=False,
        )

        registry = ToolRegistry(
            build_configured_tools(tool_config, None, None, workspace, runtime)
        )

        schema_names = [schema["name"] for schema in registry.schemas()]
        self.assertEqual(schema_names, ["glob", "grep"])

    def test_apply_patch_rejects_parent_directory_target(self) -> None:
        from gear_agent.tools.patch import ApplyPatchTool

        with tempfile.TemporaryDirectory() as temp_dir:
            tool = ApplyPatchTool(Path(temp_dir))
            patch = "\n".join(
                [
                    "--- ../outside.txt",
                    "+++ ../outside.txt",
                    "@@ -1 +1 @@",
                    "-old",
                    "+new",
                    "",
                ]
            )

            with self.assertRaises(GearError) as error:
                tool.run({"patch": patch})

            self.assertEqual(error.exception.origin, "apply_patch")


if __name__ == "__main__":
    unittest.main()
