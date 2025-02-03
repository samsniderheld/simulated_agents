import os

from typing import List
from .base_agent import BaseAgent
from .synthetic_agent import SyntheticAgent

from pydantic import BaseModel

class Txt2ImgPrompt(BaseModel):
    prompt: str

class ShotAgent(BaseAgent):
    """
    Agent that breaks a script into a series of shots and generates txt2img prompts.

    Attributes:
        script (str): The script to be broken into shots.
    """

    def __init__(self, config_file: str = None) -> None:
        """
        Initializes the ShotAgent with a configuration file.

        Args:
            config_file (str): Path to the configuration file.
        """
        super().__init__(config_file)

    def generate_shots(self, script_file: str, num_shots: int, characters: List[SyntheticAgent]) -> List[str]:
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
                  f"You are a bot that takes a script and breaks it into a series of {num_shots} text to image prompts for FLUX. "
                  "I want you to always apply this prompt structure while crafting your prompt, but make sure each prompt is a single line. "
                  "EXAMPLE PROMPT: "
                  "CMBND, "
                  f"characters: {', '.join([f'{character.name}' for character in characters])}, "
                  f"composition: Shot of {', '.join([f'{character.name}' for character in characters])}, wide shot, " 
                   "action: The two are screaming at eachother in a modern apartment. The apartment is a luxury apartment in brooklyn."
              )
            },
            {"role": "user", "content": f"Take this the following Script: {script} and convert it to precisely this number of image prompts: {num_shots}. Do not mix two prompts together. Important make sure the prompt is single line."},

        ]
        print(messages)
        response = self.llm.make_api_call(messages)
        shots = response.split('\n')
        return shots