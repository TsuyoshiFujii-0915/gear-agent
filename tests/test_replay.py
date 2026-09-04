import unittest

from gear_agent.config import ModelConfig, ReasoningReplayMode
from gear_agent.model.replay import reasoning_replay_policy


class ModelReplayScopeTests(unittest.TestCase):
    def test_redacts_configured_api_key_from_endpoint_identity(self) -> None:
        first_key = "credential-one"
        second_key = "credential-two"
        first_config = ModelConfig(
            url=(
                "https://azure.example/openai/v1/responses"
                f"?api-version=v1&api-key={first_key}"
            ),
            model="gpt-5.5",
            api_key=first_key,
            reasoning_replay=ReasoningReplayMode.ENCRYPTED,
        )
        second_config = ModelConfig(
            url=(
                "https://azure.example/openai/v1/responses"
                f"?api-version=v1&api-key={second_key}"
            ),
            model="gpt-5.5",
            api_key=second_key,
            reasoning_replay=ReasoningReplayMode.ENCRYPTED,
        )

        first_scope = reasoning_replay_policy(first_config).current_scope
        second_scope = reasoning_replay_policy(second_config).current_scope

        self.assertEqual(first_scope.endpoint_identity, second_scope.endpoint_identity)
        self.assertNotIn(first_key, first_scope.endpoint_identity)
        self.assertNotIn(second_key, second_scope.endpoint_identity)

    def test_query_semantics_remain_part_of_endpoint_identity(self) -> None:
        first_config = ModelConfig(
            url=(
                "https://azure.example/openai/v1/responses"
                "?api-version=v1"
            ),
            model="gpt-5.5",
            api_key=None,
            reasoning_replay=ReasoningReplayMode.ENCRYPTED,
        )
        second_config = ModelConfig(
            url=(
                "https://azure.example/openai/v1/responses"
                "?api-version=v2"
            ),
            model="gpt-5.5",
            api_key=None,
            reasoning_replay=ReasoningReplayMode.ENCRYPTED,
        )

        first_scope = reasoning_replay_policy(first_config).current_scope
        second_scope = reasoning_replay_policy(second_config).current_scope

        self.assertNotEqual(first_scope.endpoint_identity, second_scope.endpoint_identity)


if __name__ == "__main__":
    unittest.main()
