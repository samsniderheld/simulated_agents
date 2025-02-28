import os
import json

from google import genai
from google.genai import types
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
        
        return output_object

    @classmethod
    def from_json(cls, json_str: str):
        """
        Loads a JSON string into the ShotList object.

        Args:
            json_str (str): The JSON string to load.

        Returns:
            ShotList: The ShotList object populated with data from the JSON string.
        """
        data = json.loads(json_str)
        shots = [Shot(**shot) for shot in data["shots"]]
        return cls(shots=shots)
        

    

class LLMWrapper:
    """
    Wrapper class for different language models.

    Attributes:
        llm (str): The language model to use.
        client (OpenAI): The client for the OpenAI API.
    """

    def __init__(self, llm: str = "gemini") -> None:
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
        elif self.llm == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is not set.")
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None

    def make_api_call(self, messages: list) -> str:
        # todo: make an error catch for when the API call fails
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
        elif self.llm == "gemini":
            response = self.client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=messages[1]["content"],
                config=types.GenerateContentConfig(
                    system_instruction=messages[0]["content"]),
                )
            return response.text
        else:
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
        
        elif self.llm == "gemini":
            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=messages[1]["content"],
                config=types.GenerateContentConfig(
                    system_instruction=messages[0]["content"],
                    response_mime_type='application/json',
                    response_schema=ShotList
                    )
            )

            parsed = ShotList.from_json(response.text)
            return parsed

        return ""