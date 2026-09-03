import unittest

from gear_agent.model.responses import extract_function_calls, extract_output_text


class ResponseParsingTests(unittest.TestCase):
    def test_extracts_output_text_from_responses_message(self) -> None:
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "Hello"},
                        {"type": "output_text", "text": " world"},
                    ],
                }
            ]
        }

        text = extract_output_text(response)

        self.assertEqual(text, "Hello world")

    def test_extracts_function_call_items(self) -> None:
        response = {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "file_read",
                    "arguments": '{"path": "README.md"}',
                }
            ]
        }

        calls = extract_function_calls(response)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].call_id, "call_1")
        self.assertEqual(calls[0].name, "file_read")
        self.assertEqual(calls[0].arguments, {"path": "README.md"})


if __name__ == "__main__":
    unittest.main()
