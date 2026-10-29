from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from gear_agent.agent.events import ToolUseFinished
from gear_agent.tui import collect_chat_lines, format_progress_event
from tests.test_exa import ExaAPI, FETCH, SEARCH, URL, api, config_text, configured


def rendered_search(result: dict[str, Any]) -> tuple[str, str]:
    progress = format_progress_event(ToolUseFinished(
        session_id="s", iteration=1, call_id="search", name="web_search", result=result,
    ))
    history = collect_chat_lines([{
        "kind": "tool_result",
        "payload": {"iteration": 1, "call_id": "search", "name": "web_search", "result": result},
    }])
    assert len(history) == 1
    assert history[0].speaker == "tool"
    return progress, history[0].text


@pytest.mark.parametrize("mode", ["text", "highlights"])
@pytest.mark.parametrize("excerpt", [{}, {"highlights": None}, {"highlights": []}, {"highlights": ["", " \n "]}])
def test_exa_search_without_excerpt_keeps_title_and_url_visible(
    tmp_path: Path, api: ExaAPI, mode: str, excerpt: dict[str, object],
) -> None:
    search = SEARCH.replace('content = "highlights"', f'content = "{mode}"')
    _, tools = configured(tmp_path, config_text(search, FETCH))
    api.response = {"results": [{"title": "Example page", "url": URL,
                                 "text": "Retrieved page body", "summary": "Generated summary", **excerpt}]}
    result = tools["web_search"].run({"query": "example"})
    original = deepcopy(result)
    assert result["results"][0]["content"] is None
    assert result["results"][0]["raw_content"] == "Retrieved page body"

    for text in rendered_search(result):
        assert "tool display error" not in text
        assert "loop 1 tool web_search completed" in text
        assert "  1. Example page\n" in text
        assert f"     {URL}\n" in text
        assert "     (snippet not reported)" in text
        assert "Retrieved page body" not in text
        assert "Generated summary" not in text
    assert result == original


def test_mixed_excerpts_keep_every_search_result_visible(tmp_path: Path, api: ExaAPI) -> None:
    _, tools = configured(tmp_path, config_text(SEARCH, FETCH))
    api.response = {"results": [
        {"title": "First", "url": f"{URL}/first", "highlights": ["First excerpt"]},
        {"title": "Second", "url": f"{URL}/second", "text": "Not an excerpt"},
        {"title": "Third", "url": f"{URL}/third", "highlights": ["Third excerpt"]},
    ]}
    result = tools["web_search"].run({"query": "mixed"})

    for text in rendered_search(result):
        assert "tool display error" not in text
        assert "results 3" in text
        assert text.endswith(
            f"  1. First\n     {URL}/first\n     First excerpt\n"
            f"  2. Second\n     {URL}/second\n     (snippet not reported)\n"
            f"  3. Third\n     {URL}/third\n     Third excerpt"
        )


@pytest.mark.parametrize("snippet", ["Existing snippet", "", "x" * 170])
def test_existing_string_snippet_display_is_unchanged(snippet: str) -> None:
    result = {"query": "query", "credits": 1,
              "results": [{"title": "Page", "url": URL, "content": snippet}]}
    preview = snippet if len(snippet) <= 160 else snippet[:160] + "... [truncated]"
    expected = ("loop 1 tool web_search completed\nquery query\nresults 1\ncredits 1\n"
                f"  1. Page\n     {URL}\n     {preview}")
    assert rendered_search(result) == (expected, expected)


@pytest.mark.parametrize("fields", [{}, {"content": 1}, {"content": True},
                                    {"content": []}, {"content": {}}])
def test_missing_or_invalid_snippet_still_reports_display_error(fields: dict[str, object]) -> None:
    result = {"query": "query", "results": [{"title": "Page", "url": URL, **fields}]}
    for text in rendered_search(result):
        assert text.startswith("tool display error\nreason ")
        assert "web_search result item.content" in text
        assert "(snippet not reported)" not in text
