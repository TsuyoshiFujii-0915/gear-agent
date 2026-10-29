from __future__ import annotations

import httpx
import pytest

from gear_agent.errors import GearError
from gear_agent.tools.treg import TregSearchProvider
from tests.test_treg import SEARCH, transport_for


def test_failed_call_does_not_hide_reported_charge() -> None:
    calls: list[httpx.Request] = []
    response = httpx.Response(500, text="Upstream failure", headers={"X-Treg-Cost-Micro": "15"})
    with pytest.raises(GearError) as raised:
        TregSearchProvider(SEARCH, transport_for(response, calls)).search("query")
    assert raised.value.error_type == "treg_unexpected_charge"
    assert raised.value.details["charged_micro"] == 15
    assert raised.value.details["status"] == 500
    assert len(calls) == 2
