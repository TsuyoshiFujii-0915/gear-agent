from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from rich.console import Group, RenderableType
from rich.markdown import Markdown as RichMarkdown
from rich.padding import Padding
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Input, RichLog, Rule, Static

from gear_agent.agent.events import (
    AgentLoopEvent,
    ModelReasoningSummaryDelta,
    ModelRequestStarted,
    ModelTextDelta,
    ReasoningReplayEvaluated,
    ToolUseFinished,
    ToolUseStarted,
)
from gear_agent.agent.compaction import CompactionService
from gear_agent.agent.loop import AgentLoop
from gear_agent.config import ModelConfig, RuntimeConfig
from gear_agent.store.jsonl import JsonlContextStore
from gear_agent.tui import (
    ChatLine,
    collect_chat_lines,
    collect_token_usage,
    compact_path,
    compact_session,
    format_progress_event,
    format_tokens,
)

# Engineering-grade theme — graphite base with machined-brass and steel accents.
_BG = "#15171c"
_LINE = "#2c313b"
_TEXT = "#c7ccd6"
_MUTED = "#6c7480"
_BRASS = "#d6a052"
_STEEL = "#7ba0c4"
_TOOL = "#8b94a1"
_ERR = "#cf7060"

# Markdown accents — derived from the base theme so rendered Markdown stays on-palette.
_HEAD_1 = "#e2b878"
_STRONG = "#e6e9ef"
_EMPH = "#9db4cc"
_CODE = "#a8c08a"
_CODE_BG = "#1d2027"

# Overrides Rich's default Markdown styles, which otherwise inject high-saturation
# colours unrelated to the graphite/brass/steel theme.
_MARKDOWN_THEME = Theme(
    {
        "markdown.h1": f"bold {_HEAD_1}",
        "markdown.h2": f"bold {_BRASS}",
        "markdown.h3": f"bold {_STEEL}",
        "markdown.h4": f"bold {_MUTED}",
        "markdown.h5": f"bold {_MUTED}",
        "markdown.h6": f"bold {_MUTED}",
        "markdown.strong": f"bold {_STRONG}",
        "markdown.em": f"italic {_EMPH}",
        "markdown.code": f"{_CODE} on {_CODE_BG}",
        "markdown.link": f"underline {_STEEL}",
        "markdown.link_url": f"underline {_STEEL}",
        "markdown.block_quote": f"italic {_MUTED}",
        "markdown.item.bullet": f"bold {_BRASS}",
        "markdown.item.number": f"bold {_BRASS}",
        "markdown.hr": _LINE,
    }
)


@dataclass(frozen=True)
class _Role:
    """Visual style for a chat speaker.

    Attributes:
        label: Speaker label shown beside the accent bar.
        color: Accent colour for the bar and label.
    """

    label: str
    color: str


_ROLES: dict[str, _Role] = {
    "you": _Role("you", _STEEL),
    "assistant": _Role("gear", _BRASS),
    "tool": _Role("tool", _TOOL),
    "error": _Role("error", _ERR),
}


def _render_tool_block(text: str) -> Text:
    """Renders tool output as a muted, rail-prefixed log block.

    Tool results are a secondary channel, so they carry no accent bar and are
    dimmed below the brightness of user input and assistant output.

    Args:
        text: Tool output text, possibly spanning multiple lines.

    Returns:
        A dim Text with each line prefixed by a thin rail glyph.
    """

    block = Text()
    for index, content in enumerate(text.split("\n")):
        if index > 0:
            block.append("\n")
        block.append("┆ ", style=_LINE)
        block.append(content, style=_MUTED)
    return block


_CSS = f"""
Screen {{
    background: {_BG};
    layout: vertical;
}}

#header {{
    height: 4;
    padding: 1 3;
    background: {_BG};
}}

Rule {{
    height: 1;
    margin: 0 3;
    color: {_LINE};
    background: {_BG};
}}

#chat {{
    height: 1fr;
    padding: 1 3;
    background: {_BG};
    scrollbar-size: 1 1;
    scrollbar-color: {_LINE};
    scrollbar-color-hover: {_BRASS};
    scrollbar-background: {_BG};
}}

#model-progress {{
    display: none;
    height: auto;
    max-height: 12;
    padding: 0 5 1 5;
    background: {_BG};
    overflow-y: auto;
    scrollbar-size: 1 1;
    scrollbar-color: {_LINE};
    scrollbar-background: {_BG};
}}

#input-bar {{
    height: 1;
    margin: 1 0;
    padding: 0 3;
    background: {_BG};
    layout: horizontal;
}}

#prompt {{
    width: 4;
    background: {_BG};
    color: {_BRASS};
}}

Input {{
    height: 1;
    width: 1fr;
    border: none;
    padding: 0;
    background: {_BG};
    color: {_TEXT};
}}

Input:focus {{
    border: none;
}}
"""


class AgentProgress(Message):
    """Textual notification that queued agent progress is available."""

    def __init__(self, sink: TextualAgentLoopEventSink) -> None:
        super().__init__()
        self._sink = sink

    def drain_events(self) -> tuple[AgentLoopEvent, ...]:
        """Drains the progress events represented by this notification.

        Returns:
            Agent progress events in publication order.
        """

        return self._sink.drain_events()


class GearApp(App[None]):
    """Gear Agent interactive TUI powered by Textual."""

    CSS = _CSS

    BINDINGS = [
        Binding("ctrl+c", "quit", "quit", priority=True),
        Binding("ctrl+d", "quit", "exit"),
    ]

    def __init__(
        self,
        model: str,
        session_id: str,
        workspace: Path,
        agent_loop: AgentLoop,
        compaction: CompactionService,
        store: JsonlContextStore,
        runtime: RuntimeConfig,
        model_config: ModelConfig,
    ) -> None:
        super().__init__()
        self._model = model
        self._session_id = session_id
        self._workspace = workspace
        self._agent_loop = agent_loop
        self._compaction = compaction
        self._store = store
        self._runtime = runtime
        self._model_config = model_config
        self._initial_events = store.load(session_id)
        self._token_usage: int | None = None
        self._progress_sink: TextualAgentLoopEventSink | None = None
        self._live_iteration: int | None = None
        self._live_text_fragments: list[str] = []
        self._live_reasoning_fragments: list[str] = []

    def compose(self) -> ComposeResult:
        yield Static(self._build_header(), id="header")
        yield Rule()
        yield RichLog(id="chat", markup=True, highlight=False, wrap=True, auto_scroll=True)
        yield Static("", id="model-progress", markup=False)
        yield Rule()
        with Horizontal(id="input-bar"):
            yield Static("▌ ›", id="prompt")
            yield Input(id="input", placeholder="enter a request")

    def on_mount(self) -> None:
        self.console.push_theme(_MARKDOWN_THEME)
        self._token_usage = collect_token_usage(self._initial_events)
        self.query_one("#header", Static).update(self._build_header())
        self._render_history(collect_chat_lines(self._initial_events))
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        event.input.clear()
        if not user_text:
            return
        if user_text in {"/quit", "/exit"}:
            self.exit()
            return
        if user_text == "/compact":
            self._set_busy(True)
            self._do_compact()
            return
        self._write_message(self.query_one("#chat", RichLog), ChatLine("you", user_text))
        self._set_busy(True)
        self._do_run_agent(user_text)

    @work(thread=True)
    def _do_run_agent(self, user_text: str) -> None:
        try:
            self._agent_loop.run_turn(
                self._session_id,
                user_text,
                self._runtime.max_iterations,
                self._runtime.model_timeout_seconds,
                self._runtime.model_stream_idle_timeout_seconds,
            )
        except Exception as exc:
            self._store.append(self._session_id, "turn_error", {"text": str(exc)})
            self.call_from_thread(self._write_error, str(exc))
        self._finish_work()

    @work(thread=True)
    def _do_compact(self) -> None:
        try:
            self._compaction.compact(
                self._session_id,
                self._store,
                self._model_config,
                self._runtime.model_timeout_seconds,
                self._runtime.model_stream_idle_timeout_seconds,
            )
        except Exception as exc:
            self._store.append(self._session_id, "turn_error", {"text": str(exc)})
            self.call_from_thread(self._write_error, str(exc))
        self._finish_work()

    def _finish_work(self) -> None:
        events = self._store.load(self._session_id)
        chat_lines = collect_chat_lines(events)
        token_usage = collect_token_usage(events)
        self.call_from_thread(self._apply_result, chat_lines, token_usage)

    def _apply_result(self, chat_lines: list[ChatLine], token_usage: int | None) -> None:
        self._drain_progress_sink()
        self._reset_live_progress(None)
        self._refresh_live_progress()
        self._token_usage = token_usage
        self._render_history(chat_lines)
        self.query_one("#header", Static).update(self._build_header())
        self._set_busy(False)

    def _render_history(self, chat_lines: list[ChatLine]) -> None:
        chat = self.query_one("#chat", RichLog)
        chat.clear()
        if not chat_lines:
            chat.write(Text("▌ session ready — enter a request", style=f"italic {_MUTED}"))
            chat.write("")
        for line in chat_lines:
            self._write_message(chat, line)

    def _write_message(self, chat: RichLog, line: ChatLine) -> None:
        role = _ROLES.get(line.speaker)
        if role is None:
            raise ValueError(f"Unsupported chat speaker: {line.speaker}")
        if line.speaker == "tool":
            chat.write(Padding(_render_tool_block(line.text), (0, 0, 0, 2)))
            chat.write("")
            return
        if line.speaker == "assistant":
            body: RenderableType = RichMarkdown(line.text, code_theme="github-dark")
        else:
            body = Text(line.text, style=_TEXT)
        chat.write(
            Text.assemble(
                ("▌ ", role.color),
                (role.label, f"bold {role.color}"),
            )
        )
        chat.write(Padding(body, (0, 0, 0, 2)))
        chat.write("")

    def on_agent_progress(self, message: AgentProgress) -> None:
        self._apply_progress_events(message.drain_events())

    def _write_error(self, text: str) -> None:
        self._drain_progress_sink()
        self._reset_live_progress(None)
        self._refresh_live_progress()
        self._write_message(self.query_one("#chat", RichLog), ChatLine("error", text))

    def bind_progress_sink(self, sink: TextualAgentLoopEventSink) -> None:
        """Registers the bound sink so finalization can drain queued events.

        Args:
            sink: Progress sink bound to this app.

        Raises:
            RuntimeError: If a different sink is already registered.
        """

        if self._progress_sink is not None and self._progress_sink is not sink:
            raise RuntimeError("GearApp already has a different progress sink.")
        self._progress_sink = sink

    def _drain_progress_sink(self) -> None:
        if self._progress_sink is None:
            return
        events = self._progress_sink.drain_events()
        if len(events) > 0:
            self._apply_progress_events(events)

    def _apply_progress_events(self, events: tuple[AgentLoopEvent, ...]) -> None:
        chat = self.query_one("#chat", RichLog)
        for event in events:
            if isinstance(event, ModelRequestStarted):
                self._require_current_session(event.session_id)
                self._reset_live_progress(event.iteration)
                self._write_message(
                    chat,
                    ChatLine("tool", format_progress_event(event)),
                )
                continue
            if isinstance(event, ModelTextDelta):
                self._require_active_model_event(event.session_id, event.iteration)
                self._live_text_fragments.append(event.delta)
                continue
            if isinstance(event, ModelReasoningSummaryDelta):
                self._require_active_model_event(event.session_id, event.iteration)
                self._live_reasoning_fragments.append(event.delta)
                continue
            if isinstance(event, ToolUseStarted):
                self._require_current_session(event.session_id)
                self._reset_live_progress(None)
                self._write_message(
                    chat,
                    ChatLine("tool", format_progress_event(event)),
                )
                continue
            if isinstance(event, (ToolUseFinished, ReasoningReplayEvaluated)):
                self._require_current_session(event.session_id)
                self._write_message(
                    chat,
                    ChatLine("tool", format_progress_event(event)),
                )
                continue
            raise ValueError(f"Unsupported agent progress event: {type(event).__name__}")
        self._refresh_live_progress()

    def _require_current_session(self, session_id: str) -> None:
        if session_id != self._session_id:
            raise ValueError(
                f"Progress event session {session_id} does not match {self._session_id}."
            )

    def _require_active_model_event(self, session_id: str, iteration: int) -> None:
        self._require_current_session(session_id)
        if self._live_iteration != iteration:
            raise ValueError(
                f"Model progress iteration {iteration} has no matching active request."
            )

    def _reset_live_progress(self, iteration: int | None) -> None:
        self._live_iteration = iteration
        self._live_text_fragments.clear()
        self._live_reasoning_fragments.clear()

    def _refresh_live_progress(self) -> None:
        progress = self.query_one("#model-progress", Static)
        renderables: list[RenderableType] = []
        if len(self._live_reasoning_fragments) > 0:
            renderables.extend(
                [
                    Text("gear thinking", style=f"bold {_STEEL}"),
                    Padding(
                        Text("".join(self._live_reasoning_fragments), style=_MUTED),
                        (0, 0, 1, 2),
                    ),
                ]
            )
        if len(self._live_text_fragments) > 0:
            renderables.extend(
                [
                    Text("gear", style=f"bold {_BRASS}"),
                    Padding(
                        Text("".join(self._live_text_fragments), style=_TEXT),
                        (0, 0, 0, 2),
                    ),
                ]
            )
        progress.display = len(renderables) > 0
        progress.update(Group(*renderables) if len(renderables) > 0 else "")

    def _set_busy(self, busy: bool) -> None:
        if busy:
            self.query_one("#chat", RichLog).write(
                Text("▌ working…", style=f"italic {_BRASS}")
            )
        inp = self.query_one(Input)
        inp.disabled = busy
        if not busy:
            inp.focus()

    def _build_header(self) -> Table:
        grid = Table.grid(expand=True, padding=(0, 0))
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="right", ratio=1)
        sep = f"  [{_LINE}]│[/{_LINE}]  "
        status = sep.join(
            [
                f"[{_MUTED}]model[/{_MUTED}] [{_STEEL}]{self._model}[/{_STEEL}]",
                f"[{_MUTED}]session[/{_MUTED}] {compact_session(self._session_id)}",
                f"[{_MUTED}]tokens[/{_MUTED}] [{_BRASS}]{format_tokens(self._token_usage)}[/{_BRASS}]",
            ]
        )
        grid.add_row(f"[bold {_BRASS}]⚙ GEAR AGENT[/bold {_BRASS}]", status)
        grid.add_row(f"[{_MUTED}]{compact_path(self._workspace)}[/{_MUTED}]", "")
        return grid


class TextualAgentLoopEventSink:
    """Publishes agent loop progress events into a Textual app."""

    def __init__(self) -> None:
        self._app: GearApp | None = None
        self._events: list[AgentLoopEvent] = []
        self._notification_pending = False
        self._lock = Lock()

    def bind(self, app: GearApp) -> None:
        """Binds the sink to a running app.

        Args:
            app: Gear Agent Textual app.
        """

        if self._app is not None and self._app is not app:
            raise RuntimeError("TextualAgentLoopEventSink is already bound to another app.")
        self._app = app
        app.bind_progress_sink(self)

    def publish(self, event: AgentLoopEvent) -> None:
        """Publishes an agent loop progress event to the app.

        Args:
            event: Agent loop event.

        Raises:
            RuntimeError: If the sink is used before being bound to an app.
        """

        if self._app is None:
            raise RuntimeError("TextualAgentLoopEventSink must be bound before use.")
        should_notify = False
        with self._lock:
            self._events.append(event)
            if not self._notification_pending:
                self._notification_pending = True
                should_notify = True
        if should_notify and not self._app.post_message(AgentProgress(self)):
            with self._lock:
                self._notification_pending = False
            raise RuntimeError("Textual app rejected an agent progress notification.")

    def drain_events(self) -> tuple[AgentLoopEvent, ...]:
        """Returns and clears the currently queued progress batch.

        Returns:
            Queued events in publication order.
        """

        with self._lock:
            events = tuple(self._events)
            self._events.clear()
            self._notification_pending = False
        return events
