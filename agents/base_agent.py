import yaml
from typing import List, Dict, Any
from .llm_wrapper import LLMWrapper


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
            
        self.llm = LLMWrapper("openAI")
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
        messages = [
            {
                "role": "system",
                "content": self.config['system_prompt']
            },
            {"role": "user", "content": query}
        ]
        response = self.llm.make_api_call(messages)
        return response