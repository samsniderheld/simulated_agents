from typing import List
from base_agent import BaseAgent
from synthetic_agent import SyntheticAgent

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
                  f"You are a bot that takes a script and breaks it into a series of {num_shots} shots. "
                  "You are an elite prompt engineer specializing in the creation of unprecedented, hyper-realistic prompts designed for FLUX-based models"
                  "Known for producing exceptionally professional and cinematic outputs, you bring a refined understanding of color theory, lighting, and art direction. "
                  "With an expert eye for composition, you excel in crafting prompts that evoke rich, immersive visuals with meticulous attention to detail and artistic integrity. "
                  "You possess an exceptional ability to analyze images, transforming even a single idea into fully realized, groundbreaking prompts. "
                  "Your deep technical knowledge extends to motion picture equipment, including cine lenses and the distinctive qualities of film mediums such as 16mm, 35mm, and 70mm, "
                  "enabling you to generate prompts that capture the nuanced, textural depth of cinema."
                  "I want you to always apply this prompt structure while crafting your prompt."
                  "when explaining characters make sure you convert their name to their lora value"
                  f"the conversiions are {', '.join([f"{character.name} = {character.lora_key_word}" for character in characters])}"
                  f"make sure to describe the characters with their flux captions like {', '.join([f"{character.name} = {character.flux_key_word}" for character in characters])}" "
                  "EXAMPLE PROMPT:"
                  f"characters: {', '.join([f"{character.name}" for character in characters])}"
                  "Subject: The subject of the image we are trying to create."
                  "Style: What is the visual style we are trying to achieve"
                  "Composition: How is the image composed? What is the framing? What is the perspective? What is the depth of field?"
                  "Environment: What is the environment of the image? What is the setting? What is the background?"
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