import unittest
from langchain.tools import tool
from agentD.agents import agentD

class TestImports(unittest.TestCase):
    def test_import_agentD(self):
        @tool
        def dummy_tool(input: str) -> str:
            """A dummy tool for testing."""
            return "ok"
        agent = agentD(
            tools=[dummy_tool],
            prefix="Test prefix",
            format_instructions="Test format instructions",
            suffix="Test suffix"
        )
        self.assertIsNotNone(agent)

    def test_import_tools(self):
        from agentD.tools import generation, prediction, retrieval
        self.assertIsNotNone(generation)
        self.assertIsNotNone(prediction)
        self.assertIsNotNone(retrieval)

    def test_import_analysis(self):
        from agentD.analysis import analysis_helper, drug_likeness_analyzer
        self.assertIsNotNone(analysis_helper)
        self.assertIsNotNone(drug_likeness_analyzer)

    def test_import_config(self):
        from configs import tool_globals
        self.assertIsNotNone(tool_globals)

if __name__ == '__main__':
    unittest.main()