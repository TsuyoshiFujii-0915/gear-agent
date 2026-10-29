from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit
import tomllib

from gear_agent.errors import GearError, gear_error
from gear_agent.compaction_config import CompactionConfig, SUMMARY_COMPACTION, load_compaction_config
from gear_agent.context_budget import ContextBudgetConfig, DISABLED_CONTEXT_BUDGET


DEFAULT_DOCKER_IMAGE = "python:3.11-slim"

DEFAULT_CONFIG_TEXT = """[model]
url = "http://localhost:1234/v1/responses"
model = "local-model-id"
api_key_env = ""
reasoning_replay = "none"
stream = false

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
model_stream_idle_timeout_seconds = 60

[context_budget]
auto_compaction = false
"""


class ReasoningReplayMode(str, Enum):
    """Configured handling for opaque reasoning state."""

    NONE = "none"
    ENCRYPTED = "encrypted"


@dataclass(frozen=True)
class ModelConfig:
    """Model endpoint configuration.

    Attributes:
        url: Complete Responses API-compatible endpoint URL.
        model: Model identifier sent in the request body.
        api_key: Bearer token value, or None when no auth header should be sent.
        reasoning_replay: Explicit opaque reasoning replay mode.
        stream: Whether to request an SSE response stream.
    """

    url: str
    model: str
    api_key: str | None
    reasoning_replay: ReasoningReplayMode
    stream: bool = False

    def __post_init__(self) -> None:
        _validate_reasoning_replay_url(self.url, self.reasoning_replay)


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
        web_search: Whether to expose the web search tool.
        web_fetch: Whether to expose the web fetch tool.
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
class TavilySearchOptions:
    """Tavily-only search options, separate from common provider settings."""

    search_depth: str
    include_answer: bool
    include_raw_content: bool


@dataclass(frozen=True)
class TavilyFetchOptions:
    """Tavily-only extraction options."""

    extract_depth: str
    content_format: str
    include_images: bool
    include_favicon: bool


@dataclass(frozen=True)
class FirecrawlSearchOptions:
    """Whether Search descriptions contain query highlights instead of snippets."""

    highlights: bool


@dataclass(frozen=True)
class FirecrawlFetchOptions:
    """Explicit Cloud v2 Scrape policy; durations are in milliseconds."""

    only_main_content: bool
    max_age_ms: int
    wait_for_ms: int
    proxy: str
    store_in_cache: bool


@dataclass(frozen=True)
class WebSearchConfig:
    """Independent search provider settings with an environment-resolved secret."""

    api_key: str = field(repr=False)
    max_results: int
    timeout_seconds: int
    provider: str
    options: TavilySearchOptions | FirecrawlSearchOptions


@dataclass(frozen=True)
class WebFetchConfig:
    """Independent fetch provider settings with an environment-resolved secret."""

    api_key: str = field(repr=False)
    timeout_seconds: int
    max_content_chars: int
    provider: str
    options: TavilyFetchOptions | FirecrawlFetchOptions


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime configuration loaded from config.toml.

    Attributes:
        workdir: Workspace directory.
        session_dir: Directory for JSONL session logs.
        network_enabled: Whether sandbox shell commands can access the network.
        max_iterations: Maximum model calls per user turn.
        model_timeout_seconds: Model request timeout in seconds.
        model_stream_idle_timeout_seconds: Maximum idle time between model
            stream bytes, or None when omitted for non-stream mode.
    """

    workdir: Path
    session_dir: Path
    network_enabled: bool
    max_iterations: int
    model_timeout_seconds: int
    model_stream_idle_timeout_seconds: int | None = None


@dataclass(frozen=True)
class AppConfig:
    """Top-level Gear Agent configuration.

    Attributes:
        model: Model endpoint configuration.
        runtime: Runtime configuration.
        tool: Model-callable tool configuration.
        web_search: Independent web search configuration when enabled.
        web_fetch: Independent web fetch configuration when enabled.
        context_budget: Explicit context policy; absent legacy configuration disables it.
        compaction: Strategy policy; absent legacy configuration retains summary.
    """

    model: ModelConfig
    runtime: RuntimeConfig
    tool: ToolConfig
    web_search: WebSearchConfig | None
    web_fetch: WebFetchConfig | None
    context_budget: ContextBudgetConfig = DISABLED_CONTEXT_BUDGET
    compaction: CompactionConfig = SUMMARY_COMPACTION


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
    reasoning_replay = _load_reasoning_replay(model_table)
    stream = _load_model_stream(model_table)
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
        model_stream_idle_timeout_seconds=_load_model_stream_idle_timeout(
            runtime_table,
            stream,
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
        ModelConfig(
            url=url,
            model=model,
            api_key=api_key,
            reasoning_replay=reasoning_replay,
            stream=stream,
        ),
        runtime,
        tool,
        web_search,
        web_fetch,
        _load_context_budget(raw_data),
        load_compaction_config(raw_data, environment),
    )


def _load_context_budget(raw_data: dict[str, object]) -> ContextBudgetConfig:
    if 'context_budget' not in raw_data:
        return DISABLED_CONTEXT_BUDGET
    table = _required_table(raw_data, 'context_budget')
    _reject_unknown_keys(table, 'context_budget', {
        'auto_compaction', 'context_window_tokens', 'reserved_tokens', 'max_input_tokens',
    })
    automatic = _required_bool(table, 'auto_compaction', 'context_budget')
    window = None
    if automatic or 'context_window_tokens' in table:
        window = _required_positive_int(table, 'context_window_tokens', 'context_budget')
    reserved = 0
    if automatic or 'reserved_tokens' in table:
        value = table.get('reserved_tokens')
        if type(value) is not int or value < 0:
            raise gear_error(
                'config_value_invalid', 'context_budget.reserved_tokens must be an explicit nonnegative integer.',
                'config', True, {'table': 'context_budget', 'key': 'reserved_tokens'},
            )
        reserved = value
    maximum = None
    if 'max_input_tokens' in table:
        maximum = _required_positive_int(table, 'max_input_tokens', 'context_budget')
    return ContextBudgetConfig(automatic, window, reserved, maximum)


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


def _load_reasoning_replay(
    data: dict[str, object],
) -> ReasoningReplayMode:
    if "reasoning_replay" not in data:
        return ReasoningReplayMode.NONE
    value = _required_string(data, "reasoning_replay", "model")
    try:
        return ReasoningReplayMode(value)
    except ValueError as exc:
        raise gear_error(
            "config_value_invalid",
            (
                "Invalid value for model.reasoning_replay. "
                "Expected 'none' or 'encrypted'."
            ),
            "config",
            True,
            {
                "table": "model",
                "key": "reasoning_replay",
                "value": value,
                "expected": [
                    ReasoningReplayMode.NONE.value,
                    ReasoningReplayMode.ENCRYPTED.value,
                ],
            },
        ) from exc


def _load_model_stream(data: dict[str, object]) -> bool:
    if "stream" not in data:
        return False
    return _required_bool(data, "stream", "model")


def _load_model_stream_idle_timeout(
    data: dict[str, object],
    stream: bool,
) -> int | None:
    key = "model_stream_idle_timeout_seconds"
    if key in data:
        return _required_positive_int(data, key, "runtime")
    if stream:
        raise gear_error(
            "config_value_invalid",
            f"Missing or invalid positive integer value: runtime.{key}",
            "config",
            True,
            {"table": "runtime", "key": key},
        )
    return None


def _validate_reasoning_replay_url(
    url: str,
    mode: ReasoningReplayMode,
) -> None:
    if mode is ReasoningReplayMode.NONE:
        return
    parsed_url = urlsplit(url)
    if (
        parsed_url.username is None
        and parsed_url.password is None
        and parsed_url.fragment == ""
    ):
        return
    raise gear_error(
        "config_value_invalid",
        (
            "model.url must not contain user information or a fragment "
            "when encrypted reasoning replay is enabled."
        ),
        "config",
        True,
        {
            "table": "model",
            "key": "url",
            "reason": "replay metadata must not be derived from URL credentials",
        },
    )


def _web_provider_options(
    table: dict[str, object], name: str, common_keys: set[str], vendor_keys: set[str],
) -> tuple[str, dict[str, object]]:
    # Absence explicitly selects Tavily for legacy configuration compatibility.
    provider = _required_string(table, "provider", name) if "provider" in table else "tavily"
    if provider != "tavily":
        raise gear_error(
            "config_value_invalid", f"Unsupported provider for {name}; expected tavily.",
            "config", True, {"table": name, "key": "provider"},
        )
    _reject_unknown_keys(table, name, common_keys | vendor_keys | {"provider", "tavily"})
    if "tavily" in table:
        if vendor_keys.intersection(table):
            raise gear_error(
                "config_value_invalid", f"Cannot mix flat and nested Tavily options in {name}.",
                "config", True, {"table": name},
            )
        options = _required_table(table, "tavily")
        _reject_unknown_keys(options, f"{name}.tavily", vendor_keys)
        return provider, options
    return provider, {key: table[key] for key in vendor_keys if key in table}


def _load_web_search_config(
    raw_data: dict[str, object],
    enabled: bool,
    environment: Mapping[str, str],
) -> WebSearchConfig | None:
    if not enabled:
        return None
    table = _required_table(raw_data, "web_search")
    if table.get("provider") == "firecrawl":
        return _load_firecrawl_search_config(table, environment)
    provider, options = _web_provider_options(
        table, "web_search", {"api_key_env", "max_results", "timeout_seconds"},
        {"search_depth", "include_answer", "include_raw_content"},
    )
    vendor = TavilySearchOptions(
        search_depth=_required_web_search_depth(options),
        include_answer=_required_bool(options, "include_answer", "web_search"),
        include_raw_content=_required_bool(options, "include_raw_content", "web_search"),
    )
    max_results = _required_int_in_range(table, "max_results", "web_search", 1, 20)
    timeout = _required_positive_int(table, "timeout_seconds", "web_search")
    api_key_env = _required_string(table, "api_key_env", "web_search")
    return WebSearchConfig(
        api_key=_required_secret(api_key_env, environment, "web_search.api_key_env"),
        max_results=max_results, timeout_seconds=timeout, provider=provider, options=vendor,
    )


def _load_web_fetch_config(
    raw_data: dict[str, object],
    enabled: bool,
    environment: Mapping[str, str],
) -> WebFetchConfig | None:
    if not enabled:
        return None
    table = _required_table(raw_data, "web_fetch")
    if table.get("provider") == "firecrawl":
        return _load_firecrawl_fetch_config(table, environment)
    provider, options = _web_provider_options(
        table, "web_fetch", {"api_key_env", "timeout_seconds", "max_content_chars"},
        {"extract_depth", "content_format", "include_images", "include_favicon"},
    )
    vendor = TavilyFetchOptions(
        extract_depth=_required_web_fetch_depth(options),
        content_format=_required_web_fetch_content_format(options),
        include_images=_required_bool(options, "include_images", "web_fetch"),
        include_favicon=_required_bool(options, "include_favicon", "web_fetch"),
    )
    timeout = _required_int_in_range(table, "timeout_seconds", "web_fetch", 1, 60)
    limit = _required_positive_int(table, "max_content_chars", "web_fetch")
    api_key_env = _required_string(table, "api_key_env", "web_fetch")
    return WebFetchConfig(
        api_key=_required_secret(api_key_env, environment, "web_fetch.api_key_env"),
        timeout_seconds=timeout, max_content_chars=limit, provider=provider, options=vendor,
    )


def _firecrawl_options(
    table: dict[str, object], name: str, common_keys: set[str], defaults: dict[str, object],
) -> dict[str, object]:
    """Validates nested vendor settings and applies the documented request policy.

    Args:
        table: Enabled web tool configuration.
        name: Tool table name for diagnostics.
        common_keys: Shared fields accepted by this tool.
        defaults: Explicit Firecrawl policy for omitted vendor options.

    Returns:
        Complete vendor settings; unknown and flat vendor fields are rejected.
    """
    _reject_unknown_keys(table, name, common_keys | {"provider", "firecrawl"})
    if "firecrawl" not in table:
        return dict(defaults)
    options = _required_table(table, "firecrawl")
    _reject_unknown_keys(options, f"{name}.firecrawl", set(defaults))
    return {**defaults, **options}


def _load_firecrawl_search_config(
    table: dict[str, object], environment: Mapping[str, str],
) -> WebSearchConfig:
    options = _firecrawl_options(
        table, "web_search", {"api_key_env", "max_results", "timeout_seconds"},
        {"highlights": False},
    )
    vendor = FirecrawlSearchOptions(
        highlights=_required_bool(options, "highlights", "web_search.firecrawl"),
    )
    limit = _required_int_in_range(table, "max_results", "web_search", 1, 100)
    timeout = _required_int_in_range(table, "timeout_seconds", "web_search", 2, 301)
    key_env = _required_string(table, "api_key_env", "web_search")
    return WebSearchConfig(
        api_key=_required_secret(key_env, environment, "web_search.api_key_env"),
        max_results=limit, timeout_seconds=timeout, provider="firecrawl", options=vendor,
    )


def _load_firecrawl_fetch_config(
    table: dict[str, object], environment: Mapping[str, str],
) -> WebFetchConfig:
    options = _firecrawl_options(
        table, "web_fetch", {"api_key_env", "timeout_seconds", "max_content_chars"},
        {"only_main_content": True, "max_age_ms": 172800000, "wait_for_ms": 0,
         "proxy": "basic", "store_in_cache": False},
    )
    origin = "web_fetch.firecrawl"
    timeout = _required_int_in_range(table, "timeout_seconds", "web_fetch", 2, 301)
    age = _required_nonnegative_int(options, "max_age_ms", origin)
    wait = _required_nonnegative_int(options, "wait_for_ms", origin)
    if wait > 60000 or wait >= (timeout - 1) * 1000:
        raise gear_error(
            "config_value_invalid", "web_fetch.firecrawl.wait_for_ms must be at most 60000 and less than the API timeout.",
            "config", True, {"table": origin, "key": "wait_for_ms"},
        )
    proxy = _required_string(options, "proxy", origin)
    if proxy not in {"basic", "enhanced", "auto"}:
        raise gear_error(
            "config_value_invalid", "web_fetch.firecrawl.proxy must be basic, enhanced or auto.",
            "config", True, {"table": origin, "key": "proxy"},
        )
    vendor = FirecrawlFetchOptions(
        only_main_content=_required_bool(options, "only_main_content", origin),
        max_age_ms=age, wait_for_ms=wait, proxy=proxy,
        store_in_cache=_required_bool(options, "store_in_cache", origin),
    )
    limit = _required_positive_int(table, "max_content_chars", "web_fetch")
    key_env = _required_string(table, "api_key_env", "web_fetch")
    return WebFetchConfig(
        api_key=_required_secret(key_env, environment, "web_fetch.api_key_env"),
        timeout_seconds=timeout, max_content_chars=limit, provider="firecrawl", options=vendor,
    )


def _required_nonnegative_int(data: dict[str, object], key: str, origin: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise gear_error(
            "config_value_invalid", f"Missing or invalid nonnegative integer: {origin}.{key}",
            "config", True, {"table": origin, "key": key},
        )
    return value


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
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
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
