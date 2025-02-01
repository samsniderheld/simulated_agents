from typing import List
from base_agent import BaseAgent

class InterviewAgent(BaseAgent):
    """
    Agent that creates an interview dialogue based on context and long-term memory.

    Attributes:
        long_memory (List[str]): Long-term memory.
    """

    def __init__(self, config_file: str) -> None:
        """
        Initializes the InterviewAgent with a configuration file.

        Args:
            config_file (str): Path to the configuration file.
        """
        super().__init__(config_file)

    def create_interview(self, name: str, long_memory: List[str]) -> str:
        """
        Creates an interview dialogue based on the provided context and long-term memory.

        Args:
            long memory (List[str]): List of long memory strings.
            context (List[str]): List of context strings.

        Returns:
            str: Generated interview dialogue.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a bot that simulates a confession booth scene in reality tv shows. You take in a character name, "
                    "and the character's long-term memory. You then generate a script that simulates the character's confessiong. "
                    "The output should be in the format of a script. There should only be dialogue from the character and nothing else."
                    "It always takes place in the INT. REALITY TV SHOW CONFESSION BOOTH"
                    "The format should look like the follwoing"
                    "FADE IN:\n"
                    "INT. DRISKILL HOTEL SEMINAR ROOM - DAY\n"
                    "JOE and APRIL burst through the doors into a clean, well-lit seminar room.\n"
                    "JOE\n"
                    "Are we in time?\n"
                    "APRIL\n"
                    "How could they start without us?\n"
                    "We’re the main attraction.\n"
                    "Joe catches his breath as he leans against the podium at the front of the room.\n"
                    "JOE\n"
                    "(looking about the room)\n"
                    "We are?\n"
                    "APRIL\n"
                    "Don’t be an idiot. You know we’ve been invited to Austin to discuss script format.\n"
                    "JOE\n"
                    "But why is the room empty?\n"
                    "April and Joe look out across the room - rows of empty chairs and nary a person in sight."
                )
            },
            {"role": "user", "content":  f"Name: {name} Long-term memory: {' '.join(long_memory)}"}
        ]
        interview = self.llm.make_api_call(messages)
        return interview