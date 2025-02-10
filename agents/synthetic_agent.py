import json

from typing import List
from .base_agent import BaseAgent
from pydantic import BaseModel

class SyntheticAgent(BaseAgent):
    """
    Agent that handles synthetic memory and processing.

    Attributes:
        short_memory (List[str]): Short-term memory.
        long_memory (List[str]): Long-term memory.
    """

    def __init__(self, config_file: str) -> None:
        """
        Initializes the SyntheticAgent with a configuration file.

        Args:
            config_file (str): Path to the configuration file.
        """
        super().__init__(config_file)
        self.name = self.config["name"]
        self.lora_key_word = self.config["lora_key_word"]
        self.flux_caption = self.config["flux_caption"]
        self.base_observations = self.config["base_observations"]
        self.short_memory: List[str] = []
        self.long_memory: List[str] = []

    def add_to_memory(self, observation: str) -> None:
        """
        Adds an observation to short-term memory.

        Args:
            observation (str): The observation to add.
        """
        self.short_memory.append(observation)

    def summarize_memory(self) -> None:
        """
        Summarizes the short-term memory and appends the summary to long-term memory.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "you are a bot that takes an input paragraph about events/actions and summarizes it with a maximum of three sentences."
                    f"The response should be in the pov of the agent {self.name}"
                  ) 
            },
            {"role": "user", "content": " ".join(self.short_memory)}
        ]
        response = self.llm.make_api_call(messages)
        self.long_memory.append(response)
        self.short_memory = []

    def reflect(self) -> str:
        """
        Reflects on the long-term and short-term memory to determine feelings.

        Returns:
            str: Reflection response.
        """
        messages = [
            {
                "role": "system",
                "content": "you are a bot that takes an input text about a series of thoughts and events and determines how the subject feels about them."
            },
            {"role": "user", "content": f"{' '.join(self.long_memory)} {' '.join(self.short_memory)}"}
        ]
        response = self.llm.make_api_call(messages)
        self.long_memory.append(response)
        return response

    def process_observation(
        self, observation: str, scene: list[str], num_beats: int, current_beat: int, use_json:bool=False
    ) -> str:
        """
        Processes an observation within the given context and story beats.

        Args:
            observation (str): The observation to process.
            scene (list[str]): List of scene context strings.
            num_beats (int): Total number of story beats.
            current_beat (int): Current story beat.

        Returns:
            str: Processed observation response.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    f"story beat {current_beat} / {num_beats}, scene context: {''.join(scene)}, "
                    f"purpose: {self.config['system_prompt']} {' '.join(self.short_memory)} "
                    "Given everything we know about this character and the current scene context, "
                    "what are they doing, thinking, or saying next? Responses need to be a single sentence. "
                    "Try to wrap up the scene in the given storybeats"
                )
            },
            {"role": "user", "content": observation}
        ]
        if use_json:
            response = self.llm.make_api_call_json(messages)
            self.short_memory.append(response.to_str())
        else:
            response = self.llm.make_api_call(messages)
            self.short_memory.append(response)
        return response

    def load_observations(self, observations: List[str]) -> None:
        """
        Loads multiple observations into short-term memory.

        Args:
            observations (List[str]): List of observations to load.
        """
        self.short_memory.extend(observations)