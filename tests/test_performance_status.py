import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from textual.pilot import Pilot
from textual.widgets import Input, RichLog, Static

from gear_agent.agent.events import ToolUseStarted
from gear_agent.config import AppConfig
from gear_agent.model.arrival import ClientTime, ClientTiming, RequestTarget
from gear_agent.model.statistics import json_statistics
from gear_agent.request_events import RequestFinished, RequestScope, RequestStarted
from gear_agent.runtime import build_agent_runtime
from gear_agent.tui_app import GearApp, TextualAgentLoopEventSink
from tests.test_headless import json_response, message, model_endpoint
from tests.test_request_telemetry import configuration, records


LABELS = ("PREFILL", "DECODE", "CACHE HIT")
USAGE = {"input_tokens": 20_000, "input_tokens_details": {"cached_tokens": 16_000}}
TIMINGS = {"prompt_n": 500, "prompt_ms": 250, "predicted_per_second": 72.4}
NORMAL = ("2.0k tok/s", "72.4 tok/s", "80.0% (16.0k tok)")


def app_for(config: AppConfig) -> tuple[GearApp, TextualAgentLoopEventSink]:
    sink = TextualAgentLoopEventSink()
    runtime = build_agent_runtime(config, config.runtime, sink)
    app = GearApp(config.model.model, "s", config.runtime.workdir, runtime.loop,
                  runtime.compaction, runtime.store, config.runtime, config.model)
    sink.bind(app)
    return app, sink


def lines(app: GearApp) -> list[str]:
    widget = app.query_one("#performance-status", Static)
    return [widget.render_line(y).text.strip() for y in range(widget.size.height)]


def assert_values(app: GearApp, values: tuple[str, ...]) -> None:
    assert " │ ".join(lines(app)) == " │ ".join(
        f"{label} {value}" for label, value in zip(LABELS, values, strict=True)
    )


async def submit(app: GearApp, pilot: Pilot[None], prompt: str) -> None:
    app.query_one(Input).value = prompt
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()
    assert app.query_one(Input).has_focus


def start(config: AppConfig) -> RequestStarted:
    return RequestStarted("one", RequestScope("s", "turn-one", 1, "agent"),
                          RequestTarget.from_config(config.model), "2026-11-01T12:00:00+00:00")


def finish(request: RequestStarted, body: dict[str, Any]) -> RequestFinished:
    timing = ClientTime(1, None, None)
    return RequestFinished(request, "success", json_statistics(body),
                           ClientTiming(timing, timing, timing), None)


def test_only_matching_dispatches_change_the_three_values(tmp_path: Path) -> None:
    config = configuration(tmp_path, "http://example.test", False)
    app, sink = app_for(config)
    first = start(config)
    second = replace(first, request_id="two", scope=replace(first.scope, turn_id="next"))
    done = finish(first, {"usage": USAGE, "timings": TIMINGS})

    async def interact() -> None:
        async with app.run_test(size=(90, 20)) as pilot:
            assert_values(app, ("—",) * 3)
            sink.publish(first)
            await pilot.pause()
            assert_values(app, ("Pending",) * 3)
            sink.publish(done)
            sink.publish(ToolUseStarted("s", 1, "call", "file_read", {"path": "README.md"}))
            for purpose in ("manual_summary", "automatic_summary", "jev_summary_fallback"):
                summary = replace(first, request_id=purpose, scope=replace(first.scope, purpose=purpose))
                sink.publish(summary)
                sink.publish(replace(done, request=summary, status="provider_failure"))
            sink.publish(first)
            sink.publish(done)
            await pilot.pause()
            assert_values(app, NORMAL)
            sink.publish(second)
            sink.publish(done)
            sink.publish(first)
            foreign = replace(second, request_id="other", scope=replace(second.scope, session_id="other"))
            sink.publish(foreign)
            sink.publish(replace(done, request=foreign))
            sink.publish(replace(done, request=replace(second, scope=first.scope)))
            await pilot.pause()
            assert_values(app, ("Pending",) * 3)
            sink.publish(replace(done, request=second, status="cancelled"))
            await pilot.pause()
            assert_values(app, ("Interrupted",) * 3)

    asyncio.run(interact())


@pytest.mark.parametrize(("wire", "expected"), [
    ({"usage": {"input_tokens": 5, "input_tokens_details": {"cached_tokens": 0}},
      "timings": {"prompt_per_second": 0, "predicted_per_second": 0.01}},
     ("0.0 tok/s", "<0.1 tok/s", "0.0% (0 tok)")),
    ({"usage": {"input_tokens": 1_500_000, "input_tokens_details": {"cached_tokens": 1_500_000}}},
     ("Unavailable", "Unavailable", "100.0% (1.5M tok)")),
    ({"usage": {"input_tokens_details": {"cached_tokens": 16_000}}},
     ("Unavailable", "Unavailable", "Unavailable (16.0k tok)")),
    ({"usage": {"input_tokens": 1, "input_tokens_details": {"cached_tokens": 2}},
      "timings": {"prompt_per_second": float("nan"), "predicted_per_second": float("inf")}},
     ("Unavailable",) * 3),
    ({"timings": {"prompt_per_second": 10**400},
      "usage": {"input_tokens": 10**400, "input_tokens_details": {"cached_tokens": 1}}},
     ("1.0e+400 tok/s", "Unavailable", "Unavailable (1 tok)")),
    ({"usage": {"input_tokens": 100_000, "input_tokens_details": {"cached_tokens": 1}}},
     ("Unavailable", "Unavailable", "<0.1% (1 tok)")),
])
def test_terminal_values(tmp_path: Path, wire: dict[str, Any], expected: tuple[str, ...]) -> None:
    config = configuration(tmp_path, "http://example.test", False)
    app, sink = app_for(config)

    async def interact() -> None:
        async with app.run_test(size=(100, 20)) as pilot:
            event = start(config)
            sink.publish(event)
            sink.publish(finish(event, wire))
            await pilot.pause()
            assert_values(app, expected)

    asyncio.run(interact())


@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
def test_model_to_log_and_responsive_status(tmp_path: Path, stream: bool) -> None:
    body = {**message("Done."), "usage": USAGE, "status": "completed"}
    terminal = {"type": "response.completed", "response": body, "timings": TIMINGS}
    reply = ((200, "text/event-stream", f"data: {json.dumps(terminal)}\n\n".encode())
             if stream else json_response({**body, "timings": TIMINGS}))

    async def interact() -> None:
        with model_endpoint([reply]) as (url, requests):
            config = configuration(tmp_path, url, stream)
            app, _ = app_for(config)
            async with app.run_test(size=(80, 24)) as pilot:
                await submit(app, pilot, "Report performance")
                assert_values(app, NORMAL)
                assert len(lines(app)) == 1
                for width in (60, 36, 25, 80):
                    await pilot.resize_terminal(width, 24)
                    await pilot.pause()
                    rendered = lines(app)
                    if width == 60:
                        assert len(rendered) == 2
                    if width == 25:
                        assert "16.0k tok" in "\n".join(rendered)
                        assert "2.0k tok/s" in "\n".join(rendered)
                        assert "72.4 tok/s" in "\n".join(rendered)
                    assert " ".join(" ".join(rendered).replace("│", "").split()) == (
                        "PREFILL 2.0k tok/s DECODE 72.4 tok/s CACHE HIT 80.0% (16.0k tok)"
                    )
                    widget = app.query_one("#performance-status")
                    assert app.query_one(RichLog).region.bottom <= widget.region.y
                    assert widget.region.bottom == app.query_one(Input).region.y
                assert len(lines(app)) == 1
                row, = records(tmp_path)
                assert isinstance(app.latest_agent_request, RequestFinished)
                assert app.latest_agent_request.request.request_id == row["request_id"]
                assert row["statistics"]["prefill_rate"]["value"] == 2000
                assert row["statistics"]["decode_rate"]["value"] == 72.4
                assert row["statistics"]["cache_hit_ratio"]["value"] == 0.8
            for model, expected in [
                (replace(config.model, stream=not stream), NORMAL),
                (replace(config.model, model="new-model"), ("—",) * 3),
                (replace(config.model, url=url + "/elsewhere"), ("—",) * 3),
            ]:
                resumed, _ = app_for(replace(config, model=model))
                async with resumed.run_test(size=(90, 20)):
                    assert_values(resumed, expected)
            assert len(requests) == 1

    asyncio.run(interact())


@pytest.mark.parametrize("status", ["incomplete", "eof"])
def test_interrupted_reply_survives_resume(tmp_path: Path, status: str) -> None:
    partial = {**message("partial"), "status": "incomplete", "timings": TIMINGS}
    event = {"type": "response.incomplete", "response": partial}
    data = (f"data: {json.dumps(event)}\n\n".encode() if status == "incomplete"
            else (Path(__file__).parent / "fixtures/responses/premature_eof.sse").read_bytes())

    async def interact() -> None:
        with model_endpoint([(200, "text/event-stream", data)]) as (url, requests):
            config = configuration(tmp_path, url, True)
            app, _ = app_for(config)
            async with app.run_test() as pilot:
                await submit(app, pilot, "task")
                assert_values(app, ("Interrupted",) * 3)
            assert records(tmp_path)[0]["status"] == status
            resumed, _ = app_for(config)
            async with resumed.run_test():
                assert_values(resumed, ("Interrupted",) * 3)
            assert len(requests) == 1

    asyncio.run(interact())


def test_resume_ignores_summary_and_unrecorded_requests(tmp_path: Path) -> None:
    from gear_agent.store.jsonl import JsonlContextStore

    config = configuration(tmp_path, "http://example.test", False)
    store = JsonlContextStore(config.runtime.session_dir)
    store.append("s", "user_input", {"text": "unfinished"})
    request = start(config)
    summary = replace(request, scope=replace(request.scope, purpose="automatic_summary"))

    async def interact() -> None:
        app, _ = app_for(config)
        async with app.run_test():
            assert_values(app, ("—",) * 3)
        store.append("s", "assistant_message", {"text": "legacy response"})
        legacy, _ = app_for(config)
        async with legacy.run_test():
            assert_values(legacy, ("Unavailable",) * 3)
        store.append_request("s", finish(request, {"usage": USAGE}).payload())
        store.append_request("s", replace(finish(summary, {}), status="provider_failure").payload())
        resumed, _ = app_for(config)
        async with resumed.run_test(size=(100, 24)):
            assert_values(resumed, ("Unavailable", "Unavailable", NORMAL[2]))

    asyncio.run(interact())

@pytest.mark.parametrize(("usage", "expected"), [
    (USAGE, ("Unavailable", "Unavailable", NORMAL[2])),
    ({}, ("Unavailable",) * 3),
    ({"total_tokens": 10**400, "input_tokens_details": {"cached_tokens": 3}},
     ("Unavailable", "Unavailable", "Unavailable (3 tok)")),
    ("invalid", ("Unavailable",) * 3),
])
def test_partial_usage_keeps_chat_and_input_usable(
    tmp_path: Path, usage: object, expected: tuple[str, ...]
) -> None:
    async def interact() -> None:
        body = {**message("Visible answer"), "usage": usage}
        with model_endpoint([json_response(body)]) as (url, _):
            app, _ = app_for(configuration(tmp_path, url, False))
            async with app.run_test(size=(100, 20)) as pilot:
                await submit(app, pilot, "task")
                assert_values(app, expected)
                assert "Visible answer" in " ".join(line.text for line in app.query_one(RichLog).lines)

    asyncio.run(interact())


def test_broken_sidecar_does_not_prevent_conversation_resume(tmp_path: Path) -> None:
    config = configuration(tmp_path, "http://example.test", False)
    config.runtime.session_dir.mkdir()
    (config.runtime.session_dir / "s.requests.ndjson").write_text('{"schema_version":')
    app, _ = app_for(config)

    async def interact() -> None:
        async with app.run_test():
            assert_values(app, ("Unavailable",) * 3)
            assert app.query_one(Input).has_focus

    asyncio.run(interact())
