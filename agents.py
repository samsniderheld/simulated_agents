import json
import os
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

    def __init__(self, config_file: str) -> None:
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
                    "You are a bot that simulates a confession booth scene in reality tv shows. You take in a character name, the situation context, "
                    "and the character's long-term memory. You then generate a script that simulates the character's confessiong. "
                    "The output should be in the format of a script. There should only be dialogue from the character and nothing else. "
                )
            },
            {"role": "user", "content": f"Context: {' '.join(context)} Long-term memory: {' '.join(long_memory)}"}
        ]
        interview = self.llm.make_api_call(messages)
        return interview
    
class ShotAgent(BaseAgent):
    """
    Agent that breaks a script into a series of shots and generates txt2img prompts.

    Attributes:
        script (str): The script to be broken into shots.
    """

    def __init__(self, config_file: str) -> None:
        """
        Initializes the ShotAgent with a configuration file.

        Args:
            config_file (str): Path to the configuration file.
        """
        super().__init__(config_file)

    def generate_shots(self, script_file: str, num_shots: int) -> List[str]:
        """
        Breaks the script into a series of shots and generates txt2img prompts.

        Args:
            script_file (str): Path to the script file to be broken into shots.
            num_shots (int): The number of shots to generate.

        Returns:
            List[str]: List of txt2img prompts.
        """
        if not os.path.isfile(script_file):
            raise FileNotFoundError(f"The file {script_file} does not exist.")

        with open(script_file, 'r') as file:
            script = file.read()

        messages = [
            {
                "role": "system",
                 "content": (
                  "You are a bot that takes a script and breaks it into a series of shots. "
                  "You are an elite prompt engineer specializing in the creation of unprecedented, hyper-realistic prompts designed for FLUX-based models"
                  "Known for producing exceptionally professional and cinematic outputs, you bring a refined understanding of color theory, lighting, and art direction. "
                  "With an expert eye for composition, you excel in crafting prompts that evoke rich, immersive visuals with meticulous attention to detail and artistic integrity. "
                  "You possess an exceptional ability to analyze images, transforming even a single idea into fully realized, groundbreaking prompts. "
                  "Your deep technical knowledge extends to motion picture equipment, including cine lenses and the distinctive qualities of film mediums such as 16mm, 35mm, and 70mm, "
                  "enabling you to generate prompts that capture the nuanced, textural depth of cinema."
                  "I want you to always apply this prompt structure while crafting your prompt."
                  "EXAMPLE PROMPT:"
                  "Subject: The subject of the image we are trying to create."
                  "Style: What is the visual style we are trying to achieve"
                  "Composition: How is the image composed? What is the framing? What is the perspective? What is the depth of field?"
                  "Lighting: How is the image lit? What is the quality of the light? What is the color of the light? What is the direction of the light?"
                  "Color Palette: What is the color palette of the image? What are the dominant colors? What are the accent colors?"
                  "Mood/Atmosphere: What is the mood of the image? What is the atmosphere of the image? What is the feeling of the image?"
                  "Technical Details: What are the technical details of the image? What is the resolution? What is the aspect ratio? What is the camera type? What is the lens type?"
                  "Additional Elements: Any additional elements that should be included in the image."
                  f"extra rules : {self.config['system_prompt']}"
              )
            },
            {"role": "user", "content": f"Script: {script} Number of shots: {num_shots}"}
        ]
        response = self.llm.make_api_call(messages)
        shots = response.split('\n')
        return shots

