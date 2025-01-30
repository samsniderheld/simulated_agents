import json
from typing import List, Dict, Any
from llm_wrapper import LLMWrapper


class BaseAgent:
    """
    Base class for all agents.

    Attributes:
        config (Dict[str, Any]): Configuration loaded from the config file.
        llm (LLMWrapper): Wrapper for the language model.
        context (str): Context for the agent.
    """

    def __init__(self, config_file: str) -> None:
        """
        Initializes the BaseAgent with a configuration file.

        Args:
            config_file (str): Path to the configuration file.
        """
        self.config = self.load_config_file(config_file)
        self.llm = LLMWrapper(self.config["llm"])
        self.context = ""

    def load_config_file(self, config_file: str) -> Dict[str, Any]:
        """
        Loads the configuration file.

        Args:
            config_file (str): Path to the configuration file.

        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        with open(config_file, 'r') as file:
            config = json.load(file)
        return config


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
                "content": "you are a bot that takes an input paragraph and summarizes it with a maximum of three sentences"
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
        self, observation: str, context: str, num_beats: int, current_beat: int
    ) -> str:
        """
        Processes an observation within the given context and story beats.

        Args:
            observation (str): The observation to process.
            context (str): The context of the scene.
            num_beats (int): Total number of story beats.
            current_beat (int): Current story beat.

        Returns:
            str: Processed observation response.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    f"story beat {current_beat} / {num_beats}, scene context: {context}, "
                    f"purpose: {self.config['system_prompt']} {' '.join(self.short_memory)} "
                    "Given everything we know about this character and the current scene context, "
                    "what are they doing, thinking, or saying next? Responses need to be a single sentence. "
                    "Try to wrap up the scene in the given storybeats"
                )
            },
            {"role": "user", "content": observation}
        ]
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
            {"role": "system", "content": self.config["system_prompt"]},
            {"role": "user", "content": " .".join(context)}
        ]
        self.script = self.llm.make_api_call(messages)
        return self.script
    

class InterviewAgent(BaseAgent):
    """
    Agent that creates an interview dialogue based on context and long-term memory.

    Attributes:
        long_memory (List[str]): Long-term memory.
    """

    def __init__(self, config_file: str, long_memory: List[str]) -> None:
        """
        Initializes the InterviewAgent with a configuration file.

        Args:
            config_file (str): Path to the configuration file.
        """
        super().__init__(config_file)

    def create_interview(self, name: str, context: List[str], long_memory: List[str]) -> str:
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
                    "You are a bot that simulates an interview. You take in a character name, the situation context, "
                    "and the character's long-term memory. You then generate a script that simulates an interview with the character. "
                    "The output should be in the format of a script. The interviewer should be referred to as 'interviewer' and the character "
                    "should be referred to by their name."
                )
            },
            {"role": "user", "content": f"Context: {' '.join(context)} Long-term memory: {' '.join(long_memory)}"}
        ]
        interview = self.llm.make_api_call(messages)
        return interview