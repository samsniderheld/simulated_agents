import os
from openai import OpenAI


class LLMWrapper:
    """
    Wrapper class for different language models.

    Attributes:
        llm (str): The language model to use.
        client (OpenAI): The client for the OpenAI API.
    """

    def __init__(self, llm: str = "openAI") -> None:
        """
        Initializes the LLMWrapper with the specified language model.

        Args:
            llm (str): The language model to use. Defaults to "openAI".
        """
        self.llm = llm
        if self.llm == "openAI":
            os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
            self.client = OpenAI()
        else:
            self.client = None

    def make_api_call(self, messages: list) -> str:
        """
        Makes an API call to the language model with the provided messages.

        Args:
            messages (list): The messages to send to the language model.

        Returns:
            str: The response from the language model.
        """
        if self.llm == "openAI":
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            return response.choices[0].message.content
        return ""