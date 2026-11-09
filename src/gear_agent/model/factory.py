from collections.abc import Callable

from gear_agent.config import ModelConfig
from gear_agent.model.client import ModelClient
from gear_agent.model.responses_adapter import ResponsesModelAdapter
from gear_agent.model.transport import HttpxHttpTransport


def build_model_adapter(
    config: ModelConfig, *, on_http_status: Callable[[int], None] | None = None,
) -> ResponsesModelAdapter:
    """Constructs the supported adapter from effective model configuration.

    Args:
        config: Validated effective model configuration.
        on_http_status: Optional header-time observation for the diagnostic command.

    Returns:
        Configured Responses adapter whose callers supply per-request progress.
    """
    return ResponsesModelAdapter(ModelClient(HttpxHttpTransport(on_http_status)), config)
