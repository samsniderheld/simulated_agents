import yaml
import base64
from typing import Dict, Any
from .llm_wrapper import LLMWrapper
import numpy as np
from io import BytesIO
from PIL import Image


class BaseAgent:
    """
    Base class for all agents.

    Attributes:
        config (Dict[str, Any]): Configuration loaded from the config file.
        llm (LLMWrapper): Wrapper for the language model.
        context (str): Context for the agent.
    """

    def __init__(self, config_file: str = None) -> None:
        """
        Initializes the BaseAgent with a configuration file.

        Args:
            config_file (str): Path to the configuration file. Defaults to None.
        """
        if config_file:
            self.config = self.load_config_file(config_file)
        else:
            self.config = self.default_config()
            
        self.llm = LLMWrapper("gemini")
        self.name = self.config["name"]

    def load_config_file(self, config_file: str) -> Dict[str, Any]:
        """
        Loads the configuration file.

        Args:
            config_file (str): Path to the configuration file.

        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def default_config(self) -> Dict[str, Any]:
        """
        Provides a default configuration.

        Returns:
            Dict[str, Any]: Default configuration dictionary.
        """
        return {
            "system_prompt": "Default system prompt.",
            "llm": "openAI"
        }

    def basic_api_call(self, query: str) -> str:
        """
        Makes a basic API call to the language model with the provided query.

        Args:
            query (str): The query to send to the language model.

        Returns:
            str: The response from the language model.
        """
        messages = [
            {
                "role": "system",
                "content": self.config['system_prompt']
            },
            {"role": "user", "content": query}
        ]
        response = self.llm.make_api_call(messages)
        return response

    def basic_api_call_structured(self, query: str) -> str:
        """
        Makes a basic API call to the language model with the provided query and expects a structured response.

        Args:
            query (str): The query to send to the language model.

        Returns:
            str: The structured response from the language model.
        """
        messages = [
            {
                "role": "system",
                "content": self.config['system_prompt']
            },
            {"role": "user", "content": query}
        ]
        response = self.llm.make_api_call_structured(messages)
        return response
    
    def image_api_call(self, query: str, image:Image) -> str:
        """
        Makes an API call to the language model with the provided query and image for prompt generation.

        Args:
            query (str): The query to send to the language model.

        Returns:
            str: The response from the language model.
        """

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        messages = [
            {
                "role": "system",
                "content": self.config['system_prompt']
            },
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": query,
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url":  f"data:image/jpeg;base64,{base64_image}"
                    },
                    },
                ],
            }
        ]
        response = self.llm.make_api_call(messages)
        return response