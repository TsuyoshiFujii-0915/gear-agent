import importlib.util
import unittest
from importlib.metadata import distribution


class ProjectIdentityTests(unittest.TestCase):
    def test_exposes_only_gear_agent_python_package(self) -> None:
        self.assertIsNotNone(importlib.util.find_spec("gear_agent"))
        self.assertIsNone(importlib.util.find_spec("gear_code"))

    def test_distribution_is_named_gear_agent(self) -> None:
        package = distribution("gear-agent")

        self.assertEqual(package.metadata["Name"], "gear-agent")


if __name__ == "__main__":
    unittest.main()
