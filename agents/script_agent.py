from typing import List
from .base_agent import BaseAgent

class ScriptAgent(BaseAgent):
    """
    Agent that writes scripts based on context.
    """

    def write_script(self, context: List[str]) -> str:
        """
        Writes a script based on the provided context list.

        Args:
            context (List[str]): List of context strings.

        Returns:
            str: Generated script.
        """
        messages = [
            {"role": "system", "content": (
                "You are a script writer. You adapt a list of observations and dialouge in to a script."
                "The script can be as long as it needs to be. It should be formated as a script with dialogue, scene descriptions, and actions. "
                "make sure the script follows a format like the one below: "
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
            {"role": "user", "content": " .".join(context)}
        ]
        self.script = self.llm.make_api_call(messages)
        return self.script