from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import tomllib

from gear_agent.errors import GearError, gear_error


DEFAULT_DOCKER_IMAGE = "python:3.11-slim"

DEFAULT_CONFIG_TEXT = """[model]
url = "http://localhost:1234/v1/responses"
model = "local-model-id"
api_key_env = ""

[tool]
shell_tool = true
file_read = true
file_write = true
apply_patch = true
glob = true
grep = true
web_search = false
web_fetch = false

[web_search]
api_key_env = "TAVILY_API_KEY"
search_depth = "basic"
max_results = 5
timeout_seconds = 20
include_answer = true
include_raw_content = false

[web_fetch]
api_key_env = "TAVILY_API_KEY"
extract_depth = "basic"
content_format = "markdown"
timeout_seconds = 20
include_images = false
include_favicon = true
max_content_chars = 20000

[runtime]
workdir = "."
session_dir = ".gear/sessions"
network = "disabled"
max_iterations = 8
model_timeout_seconds = 120
"""


@dataclass(frozen=True)
class ModelConfig:
    """Model endpoint configuration.

    Attributes:
        url: Complete Responses API-compatible endpoint URL.
        model: Model identifier sent in the request body.
        api_key: Bearer token value, or None when no auth header should be sent.
    """

    url: str
    model: str
    api_key: str | None


@dataclass(frozen=True)
class ToolConfig:
    """Model-callable tool configuration.

    Attributes:
        shell_tool: Whether to expose the shell execution tool.
        file_read: Whether to expose the file read tool.
        file_write: Whether to expose the file write tool.
        apply_patch: Whether to expose the patch application tool.
        glob: Whether to expose the filesystem glob tool.
        grep: Whether to expose the filesystem grep tool.
        web_search: Whether to expose the Tavily web search tool.
        web_fetch: Whether to expose the Tavily web fetch tool.
    """

    shell_tool: bool
    file_read: bool
    file_write: bool
    apply_patch: bool
    glob: bool
    grep: bool
    web_search: bool
    web_fetch: bool


@dataclass(frozen=True)
class WebSearchConfig:
    """Tavily web search configuration.

    Attributes:
        api_key: Tavily API key resolved from the configured environment variable.
        search_depth: Tavily search depth.
        max_results: Maximum Tavily search results to return.
        timeout_seconds: Tavily request timeout in seconds.
        include_answer: Whether Tavily should include a generated answer.
        include_raw_content: Whether Tavily should include raw result content.
    """

    api_key: str
    search_depth: str
    max_results: int
    timeout_seconds: int
    include_answer: bool
    include_raw_content: bool


@dataclass(frozen=True)
class WebFetchConfig:
    """Tavily web fetch configuration.

    Attributes:
        api_key: Tavily API key resolved from the configured environment variable.
        extract_depth: Tavily extraction depth.
        content_format: Extracted content format.
        timeout_seconds: Tavily request timeout in seconds.
        include_images: Whether Tavily should include page image URLs.
        include_favicon: Whether Tavily should include the page favicon URL.
        max_content_chars: Maximum content length accepted from Tavily.
    """

    api_key: str
    extract_depth: str
    content_format: str
    timeout_seconds: int
    include_images: bool
    include_favicon: bool
    max_content_chars: int


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime configuration loaded from config.toml.

    Attributes:
        workdir: Workspace directory.
        session_dir: Directory for JSONL session logs.
        network_enabled: Whether sandbox shell commands can access the network.
        max_iterations: Maximum model calls per user turn.
        model_timeout_seconds: Model request timeout in seconds.
    """

    workdir: Path
    session_dir: Path
    network_enabled: bool
    max_iterations: int
    model_timeout_seconds: int


@dataclass(frozen=True)
class AppConfig:
    """Top-level Gear Agent configuration.

    Attributes:
        model: Model endpoint configuration.
        runtime: Runtime configuration.
        tool: Model-callable tool configuration.
        web_search: Tavily web search configuration when enabled.
        web_fetch: Tavily web fetch configuration when enabled.
    """

    model: ModelConfig
    runtime: RuntimeConfig
    tool: ToolConfig
    web_search: WebSearchConfig | None
    web_fetch: WebFetchConfig | None


def load_config(path: Path, environment: Mapping[str, str]) -> AppConfig:
    """Loads Gear Agent configuration from TOML.

    Args:
        path: Path to the TOML configuration file.
        environment: Environment mapping used to resolve API keys.

    Returns:
        Parsed application configuration.

    Raises:
        GearError: If the file cannot be read or required values are invalid.
    """

    try:
        raw_data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise gear_error(
            "config_read_failed",
            f"Failed to read config file: {path}",
            "config",
            True,
            {"path": str(path), "reason": str(exc)},
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise gear_error(
            "config_parse_failed",
            f"Failed to parse config file: {path}",
            "config",
            True,
            {"path": str(path), "reason": str(exc)},
        ) from exc

    model_table = _required_table(raw_data, "model")
    url = _required_string(model_table, "url", "model")
    model = _required_string(model_table, "model", "model")
    api_key_env = _required_string(model_table, "api_key_env", "model")
    api_key = _resolve_api_key(api_key_env, environment)
    runtime_table = _required_table(raw_data, "runtime")
    runtime = RuntimeConfig(
        workdir=Path(_required_string(runtime_table, "workdir", "runtime")),
        session_dir=Path(_required_string(runtime_table, "session_dir", "runtime")),
        network_enabled=_required_network(runtime_table),
        max_iterations=_required_positive_int(runtime_table, "max_iterations", "runtime"),
        model_timeout_seconds=_required_positive_int(
            runtime_table,
            "model_timeout_seconds",
            "runtime",
        ),
    )
    tool_table = _required_table(raw_data, "tool")
    _reject_unknown_keys(
        tool_table,
        "tool",
        {
            "shell_tool",
            "file_read",
            "file_write",
            "apply_patch",
            "glob",
            "grep",
            "web_search",
            "web_fetch",
        },
    )
    tool = ToolConfig(
        shell_tool=_required_bool(tool_table, "shell_tool", "tool"),
        file_read=_required_bool(tool_table, "file_read", "tool"),
        file_write=_required_bool(tool_table, "file_write", "tool"),
        apply_patch=_required_bool(tool_table, "apply_patch", "tool"),
        glob=_required_bool(tool_table, "glob", "tool"),
        grep=_required_bool(tool_table, "grep", "tool"),
        web_search=_required_bool(tool_table, "web_search", "tool"),
        web_fetch=_required_bool(tool_table, "web_fetch", "tool"),
    )
    web_search = _load_web_search_config(raw_data, tool.web_search, environment)
    web_fetch = _load_web_fetch_config(raw_data, tool.web_fetch, environment)
    return AppConfig(
        ModelConfig(url=url, model=model, api_key=api_key),
        runtime,
        tool,
        web_search,
        web_fetch,
    )


def discover_config_path(start_dir: Path, home_dir: Path) -> Path:
    """Finds the effective Gear Agent config file.

    Project-scoped config files are searched first, walking from start_dir
    toward the filesystem root. User-scoped config is used only when no
    project config exists.

    Args:
        start_dir: Directory used as the project discovery starting point.
        home_dir: User home directory.

    Returns:
        Path to the selected config file.

    Raises:
        GearError: If no config file exists in either scope.
    """

    current = start_dir.absolute()
    for directory in [current, *current.parents]:
        candidate = directory / ".gear" / "config.toml"
        if candidate.is_file():
            return candidate

    user_candidate = user_config_path(home_dir)
    if user_candidate.is_file():
        return user_candidate

    raise gear_error(
        "config_not_found",
        "No Gear Agent config found. Run 'gear init' or 'gear init --scope user'.",
        "config",
        True,
        {
            "project_candidate": str(project_config_path(start_dir)),
            "user_candidate": str(user_candidate),
        },
    )


def initialize_config(scope: str, project_root: Path, home_dir: Path) -> Path:
    """Creates a default config file for a scope.

    Args:
        scope: Either "project" or "user".
        project_root: Project root used for project-scoped config.
        home_dir: Home directory used for user-scoped config.

    Returns:
        Created config path.

    Raises:
        GearError: If scope is invalid or the target already exists.
    """

    if scope == "project":
        path = project_config_path(project_root)
    elif scope == "user":
        path = user_config_path(home_dir)
    else:
        raise gear_error(
            "config_scope_invalid",
            "Invalid config scope. Expected 'project' or 'user'.",
            "config",
            True,
            {"scope": scope},
        )

    if path.exists():
        raise gear_error(
            "config_already_exists",
            f"Config already exists: {path}",
            "config",
            True,
            {"path": str(path)},
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_CONFIG_TEXT, encoding="utf-8")
    except OSError as exc:
        raise gear_error(
            "config_write_failed",
            f"Failed to write config file: {path}",
            "config",
            True,
            {"path": str(path), "reason": str(exc)},
        ) from exc
    return path


def project_config_path(project_root: Path) -> Path:
    """Returns the project-scoped config path.

    Args:
        project_root: Project root directory.

    Returns:
        Project-scoped config path.
    """

    return project_root.absolute() / ".gear" / "config.toml"


def user_config_path(home_dir: Path) -> Path:
    """Returns the user-scoped config path.

    Args:
        home_dir: User home directory.

    Returns:
        User-scoped config path.
    """

    return home_dir.absolute() / ".gear" / "config.toml"


def _required_table(data: dict[str, object], key: str) -> dict[str, object]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise gear_error(
            "config_value_invalid",
            f"Missing or invalid [{key}] table.",
            "config",
            True,
            {"key": key},
        )
    return value


def _required_string(data: dict[str, object], key: str, table_name: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise gear_error(
            "config_value_invalid",
            f"Missing or invalid string value: {table_name}.{key}",
            "config",
            True,
            {"table": table_name, "key": key},
        )
    return value


def _resolve_api_key(api_key_env: str, environment: Mapping[str, str]) -> str | None:
    if api_key_env == "":
        return None
    value = environment.get(api_key_env)
    if value is None or value == "":
        raise gear_error(
            "config_secret_missing",
            f"Environment variable is required but not set: {api_key_env}",
            "config",
            True,
            {"env": api_key_env},
        )
    return value


def _required_network(data: dict[str, object]) -> bool:
    value = _required_string(data, "network", "runtime")
    if value == "enabled":
        return True
    if value == "disabled":
        return False
    raise gear_error(
        "config_value_invalid",
        "Invalid value for runtime.network. Expected 'enabled' or 'disabled'.",
        "config",
        True,
        {"table": "runtime", "key": "network", "value": value},
    )


def _load_web_search_config(
    raw_data: dict[str, object],
    enabled: bool,
    environment: Mapping[str, str],
) -> WebSearchConfig | None:
    if not enabled:
        return None
    table = _required_table(raw_data, "web_search")
    _reject_unknown_keys(
        table,
        "web_search",
        {
            "api_key_env",
            "search_depth",
            "max_results",
            "timeout_seconds",
            "include_answer",
            "include_raw_content",
        },
    )
    api_key_env = _required_string(table, "api_key_env", "web_search")
    api_key = _required_secret(api_key_env, environment, "web_search.api_key_env")
    return WebSearchConfig(
        api_key=api_key,
        search_depth=_required_web_search_depth(table),
        max_results=_required_int_in_range(table, "max_results", "web_search", 1, 20),
        timeout_seconds=_required_positive_int(table, "timeout_seconds", "web_search"),
        include_answer=_required_bool(table, "include_answer", "web_search"),
        include_raw_content=_required_bool(table, "include_raw_content", "web_search"),
    )


def _load_web_fetch_config(
    raw_data: dict[str, object],
    enabled: bool,
    environment: Mapping[str, str],
) -> WebFetchConfig | None:
    if not enabled:
        return None
    table = _required_table(raw_data, "web_fetch")
    _reject_unknown_keys(
        table,
        "web_fetch",
        {
            "api_key_env",
            "extract_depth",
            "content_format",
            "timeout_seconds",
            "include_images",
            "include_favicon",
            "max_content_chars",
        },
    )
    api_key_env = _required_string(table, "api_key_env", "web_fetch")
    api_key = _required_secret(api_key_env, environment, "web_fetch.api_key_env")
    return WebFetchConfig(
        api_key=api_key,
        extract_depth=_required_web_fetch_depth(table),
        content_format=_required_web_fetch_content_format(table),
        timeout_seconds=_required_positive_int(table, "timeout_seconds", "web_fetch"),
        include_images=_required_bool(table, "include_images", "web_fetch"),
        include_favicon=_required_bool(table, "include_favicon", "web_fetch"),
        max_content_chars=_required_positive_int(table, "max_content_chars", "web_fetch"),
    )


def _required_secret(
    api_key_env: str,
    environment: Mapping[str, str],
    origin: str,
) -> str:
    if api_key_env == "":
        raise gear_error(
            "config_secret_env_invalid",
            f"Environment variable name is required: {origin}",
            "config",
            True,
            {"origin": origin},
        )
    value = environment.get(api_key_env)
    if value is None or value == "":
        raise gear_error(
            "config_secret_missing",
            f"Environment variable is required but not set: {api_key_env}",
            "config",
            True,
            {"env": api_key_env, "origin": origin},
        )
    return value


def _required_web_search_depth(data: dict[str, object]) -> str:
    value = _required_string(data, "search_depth", "web_search")
    expected_values = {"advanced", "basic", "fast", "ultra-fast"}
    if value not in expected_values:
        raise gear_error(
            "config_value_invalid",
            "Invalid value for web_search.search_depth.",
            "config",
            True,
            {
                "table": "web_search",
                "key": "search_depth",
                "value": value,
                "expected": sorted(expected_values),
            },
        )
    return value


def _required_web_fetch_depth(data: dict[str, object]) -> str:
    value = _required_string(data, "extract_depth", "web_fetch")
    expected_values = {"advanced", "basic"}
    if value not in expected_values:
        raise gear_error(
            "config_value_invalid",
            "Invalid value for web_fetch.extract_depth.",
            "config",
            True,
            {
                "table": "web_fetch",
                "key": "extract_depth",
                "value": value,
                "expected": sorted(expected_values),
            },
        )
    return value


def _required_web_fetch_content_format(data: dict[str, object]) -> str:
    value = _required_string(data, "content_format", "web_fetch")
    expected_values = {"markdown", "text"}
    if value not in expected_values:
        raise gear_error(
            "config_value_invalid",
            "Invalid value for web_fetch.content_format.",
            "config",
            True,
            {
                "table": "web_fetch",
                "key": "content_format",
                "value": value,
                "expected": sorted(expected_values),
            },
        )
    return value


def _required_int_in_range(
    data: dict[str, object],
    key: str,
    table_name: str,
    minimum: int,
    maximum: int,
) -> int:
    value = _required_positive_int(data, key, table_name)
    if value < minimum or value > maximum:
        raise gear_error(
            "config_value_invalid",
            f"Invalid integer range for {table_name}.{key}.",
            "config",
            True,
            {
                "table": table_name,
                "key": key,
                "value": value,
                "minimum": minimum,
                "maximum": maximum,
            },
        )
    return value


def _required_positive_int(data: dict[str, object], key: str, table_name: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or value < 1:
        raise gear_error(
            "config_value_invalid",
            f"Missing or invalid positive integer value: {table_name}.{key}",
            "config",
            True,
            {"table": table_name, "key": key},
        )
    return value


def _required_bool(data: dict[str, object], key: str, table_name: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise gear_error(
            "config_value_invalid",
            f"Missing or invalid boolean value: {table_name}.{key}",
            "config",
            True,
            {"table": table_name, "key": key},
        )
    return value


def _reject_unknown_keys(
    data: dict[str, object],
    table_name: str,
    expected_keys: set[str],
) -> None:
    for key in data:
        if key not in expected_keys:
            raise gear_error(
                "config_value_invalid",
                f"Unknown config value: {table_name}.{key}",
                "config",
                True,
                {"table": table_name, "key": key},
            )
