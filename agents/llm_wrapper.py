import os
import json
from openai import OpenAI
from pydantic import BaseModel

class Shot(BaseModel):
    shot_action: str
    txt2img_prompt: str
    vo:str

class ShotList(BaseModel):
    shots: list[Shot]

    def to_str(self):
        output_string = ""
        for i,shot in enumerate(self.shots):
            output_string+=f"shot {i} action: {shot.shot_action}\n"
            output_string+=f"shot {i} prompt: {shot.txt2img_prompt}\n"
            output_string+=f"shot {i} vo: {shot.vo}\n"
        
        return output_string
    
    def to_json(self):
        output_object = {"shots":[]}
        for i,shot in enumerate(self.shots):
            output_object["shots"].append({"shot_action":shot.shot_action, "txt2img_prompt":shot.txt2img_prompt, "vo":shot.vo})
        
        return json.dump(output_object,indent=4)

    

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
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set.")
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
                model="gpt-4o",
                messages=messages
            )
            return response.choices[0].message.content
        return ""
    
    def make_api_call_structured(self, messages: list) -> str:
        """
        Makes an API call to the language model with the provided messages.

        Args:
            messages (list): The messages to send to the language model.

        Returns:
            str: The the structured response from the language model.
        """
        if self.llm == "openAI":
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=messages,
                response_format=ShotList
                
            )
            return response.choices[0].message.parsed
        return ""