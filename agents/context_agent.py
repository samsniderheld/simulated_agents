from typing import List
from .base_agent import BaseAgent

class ContextAgent(BaseAgent):
    """
    Agent that updates and manages context.
    """

    def update_context(self, context: List[str]) -> str:
        """
        Updates the context based on the provided context list.

        Args:
            context (List[str]): List of context strings.

        Returns:
            str: Updated context.
        """
        messages = [
            {"role": "system", "content": self.config["system_prompt"]},
            {"role": "user", "content": " .".join(context)}
        ]
        self.context = self.llm.make_api_call(messages)
        return self.context