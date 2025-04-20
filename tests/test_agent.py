import unittest
from agent.agent import CodeAgent

class TestCodeAgent(unittest.TestCase):
    def test_agent_run(self):
        agent = CodeAgent()
        user_query = "A simple query. Determine subgoals and execute code."
        result = agent.run(user_query)
        self.assertIn("final_answer", result)
        self.assertIn("plan", result)

if __name__ == '__main__':
    unittest.main() 