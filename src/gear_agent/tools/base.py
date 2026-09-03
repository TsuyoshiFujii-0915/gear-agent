from __future__ import annotations

from abc import ABC, abstractmethod


class Tool(ABC):
    """Executable tool exposed to the model."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used by function calling."""

    @abstractmethod
    def schema(self) -> dict[str, object]:
        """Returns the Responses API function tool schema."""

    @abstractmethod
    def run(self, arguments: dict[str, object]) -> dict[str, object]:
        """Runs the tool.

        Args:
            arguments: Parsed tool call arguments.

        Returns:
            JSON-serializable tool result.
        """
