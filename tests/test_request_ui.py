import asyncio
from pathlib import Path

from textual.widgets import Input

from gear_agent.request_events import RequestFinished
from gear_agent.runtime import build_agent_runtime
from gear_agent.tui_app import GearApp, TextualAgentLoopEventSink
from tests.test_headless import json_response, message, model_endpoint
from tests.test_request_telemetry import configuration, records


def test_tui_keeps_latest_agent_request_across_compact_and_delayed_finish(tmp_path: Path) -> None:
    async def interact() -> None:
        with model_endpoint([json_response(message('done'))] * 3) as (url, requests):
            config = configuration(tmp_path, url, False)
            sink = TextualAgentLoopEventSink()
            runtime = build_agent_runtime(config, config.runtime, sink)
            app = GearApp(config.model.model, 's', tmp_path, runtime.loop, runtime.compaction,
                          runtime.store, config.runtime, config.model)
            sink.bind(app)
            async with app.run_test() as pilot:
                replies: list[RequestFinished] = []
                for prompt in ('task', 'next', '/compact'):
                    app.query_one(Input).value = prompt
                    await pilot.press('enter')
                    await app.workers.wait_for_complete()
                    await pilot.pause()
                    event = app.latest_agent_request
                    assert isinstance(event, RequestFinished)
                    replies.append(event)
                sink.publish(replies[0])
                await pilot.pause()
                assert app.latest_agent_request == replies[1] == replies[2]
            rows = records(tmp_path)
            assert len(rows) == len(requests) == 3
            assert [event.request.request_id for event in replies[:2]] == [row['request_id'] for row in rows[:2]]
            assert rows[-1]['purpose'] == 'manual_summary'

    asyncio.run(interact())


def test_storage_diagnostic_reaches_interactive_sink_and_preserves_answer(tmp_path: Path) -> None:
    from tests.test_request_telemetry import Events

    with model_endpoint([json_response(message('done'))]) as (url, requests):
        config = configuration(tmp_path, url, False)
        sink = Events()
        runtime = build_agent_runtime(config, config.runtime, sink)
        (config.runtime.session_dir / 's.requests.ndjson').mkdir()
        assert runtime.loop.run_turn('s', 'task', 1, 5).final_text == 'done'
        diagnostic, = [event for event in sink.received if type(event).__name__ == 'RequestLogWriteFailed']
        finished, = [event for event in sink.received if isinstance(event, RequestFinished)]
        assert diagnostic.request_id == finished.request.request_id
        assert [event['kind'] for event in runtime.store.load('s')] == ['user_input', 'model_response', 'assistant_message']
        assert len(requests) == 1
