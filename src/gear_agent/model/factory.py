from gear_agent.config import ModelConfig
from gear_agent.model.adapter import ModelAdapter
from gear_agent.model.client import ModelClient
from gear_agent.model.responses_adapter import ResponsesModelAdapter
from gear_agent.model.transport import HttpxHttpTransport


def build_model_adapter(config: ModelConfig) -> ModelAdapter:
    """Constructs the supported adapter from effective model configuration.

    Args:
        config: Validated effective model configuration.

    Returns:
        Configured Responses adapter with the existing silent progress behavior.
    """
    return ResponsesModelAdapter(ModelClient(HttpxHttpTransport()), config)
