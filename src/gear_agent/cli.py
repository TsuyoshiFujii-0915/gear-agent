from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Mapping
from pathlib import Path
from uuid import uuid4
import os
import sys

from gear_agent.agent.compaction import CompactionService
from gear_agent.agent.loop import AgentLoop
from gear_agent.config import (
    DEFAULT_DOCKER_IMAGE,
    RuntimeConfig,
    discover_config_path,
    initialize_config,
    load_config,
)
from gear_agent.errors import GearError
from gear_agent.model.client import ModelClient
from gear_agent.model.transport import UrllibHttpTransport
from gear_agent.store.jsonl import JsonlContextStore
from gear_agent.tools.configured import build_configured_tools
from gear_agent.tools.runtimes import DockerShellRuntime
from gear_agent.tui_app import GearApp, TextualAgentLoopEventSink


def main() -> None:
    """Entrypoint for the gear command."""

    exit_code = run_cli(sys.argv[1:], os.environ)
    raise SystemExit(exit_code)


def run_cli(argv: list[str], environment: Mapping[str, str]) -> int:
    """Runs Gear Agent command-line entrypoint.

    Args:
        argv: Command-line arguments without program name.
        environment: Environment variables.

    Returns:
        Process exit code.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            _run_init(args, environment)
        else:
            _run_tui(args, environment)
    except GearError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="gear")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to gear TOML config. Overrides scoped config discovery.",
    )
    parser.add_argument("--workdir", type=Path, help="Override runtime.workdir.")
    parser.add_argument("--session-dir", type=Path, help="Override runtime.session_dir.")
    parser.add_argument(
        "--network",
        choices=["enabled", "disabled"],
        help="Override runtime.network.",
    )
    parser.add_argument("--max-iterations", type=int, help="Override runtime.max_iterations.")
    parser.add_argument(
        "--model-timeout-seconds",
        type=int,
        help="Override runtime.model_timeout_seconds.",
    )
    subparsers = parser.add_subparsers(dest="command")
    init_parser = subparsers.add_parser("init", help="Create a Gear Agent config file.")
    init_parser.add_argument(
        "--scope",
        choices=["project", "user"],
        default="project",
        help="Config scope to initialize. Defaults to project.",
    )
    return parser


def _run_tui(args: Namespace, environment: Mapping[str, str]) -> None:
    config_path = args.config
    if config_path is None:
        config_path = discover_config_path(Path.cwd(), _home_dir(environment))
    config = load_config(config_path, environment)
    runtime = _runtime_from_args(config.runtime, args)
    workspace = runtime.workdir.resolve()
    session_id = str(uuid4())
    shell_runtime = DockerShellRuntime(workspace, DEFAULT_DOCKER_IMAGE, runtime.network_enabled)
    tools = build_configured_tools(
        config.tool,
        config.web_search,
        config.web_fetch,
        workspace,
        shell_runtime,
    )
    store = JsonlContextStore(runtime.session_dir)
    event_sink = TextualAgentLoopEventSink()
    loop = AgentLoop(
        ModelClient(UrllibHttpTransport()),
        config.model,
        tools,
        store,
        event_sink,
    )
    compaction = CompactionService(ModelClient(UrllibHttpTransport()))
    app = GearApp(
        model=config.model.model,
        session_id=session_id,
        workspace=workspace,
        agent_loop=loop,
        compaction=compaction,
        store=store,
        runtime=runtime,
        model_config=config.model,
    )
    event_sink.bind(app)
    app.run()


def _run_init(args: Namespace, environment: Mapping[str, str]) -> None:
    path = initialize_config(args.scope, Path.cwd(), _home_dir(environment))
    print(f"created> {path}")


def _runtime_from_args(runtime: RuntimeConfig, args: Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        workdir=args.workdir if args.workdir is not None else runtime.workdir,
        session_dir=args.session_dir if args.session_dir is not None else runtime.session_dir,
        network_enabled=_network_enabled_from_arg(args.network, runtime.network_enabled),
        max_iterations=(
            args.max_iterations if args.max_iterations is not None else runtime.max_iterations
        ),
        model_timeout_seconds=(
            args.model_timeout_seconds
            if args.model_timeout_seconds is not None
            else runtime.model_timeout_seconds
        ),
    )


def _network_enabled_from_arg(network: str | None, config_value: bool) -> bool:
    if network is None:
        return config_value
    return network == "enabled"


def _home_dir(environment: Mapping[str, str]) -> Path:
    home = environment.get("HOME")
    if home is None or home == "":
        raise GearError(
            "home_missing",
            "HOME environment variable is required for user-scoped config discovery.",
            "config",
            True,
            {},
        )
    return Path(home)


if __name__ == "__main__":
    main()
