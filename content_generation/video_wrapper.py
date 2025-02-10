import base64
import numpy as np
import os
import time
import urllib.request

from PIL import Image
from runwayml import RunwayML

class VideoWrapper:
    """
    A wrapper class for the RunwayML API to generate videos from images using a pre-trained model.

    Attributes:
        api (str): The API to use for video generation.
        poll_rate (int): The rate at which to poll the API for task status.
        client (RunwayML): The RunwayML client object.
    """

    def __init__(self, api: str = "runway", poll_rate: int = 10) -> None:
        """
        Initializes the VideoWrapper with the specified API and poll rate.

        Args:
            api (str): The API to use for video generation. Defaults to "runway".
            poll_rate (int): The rate at which to poll the API for task status. Defaults to 10 seconds.
        """
        self.api = api
        self.poll_rate = poll_rate 
        if self.api == "runway":
            api_key = os.getenv("RUNWAYML_API_SECRET")
            if not api_key:
                raise ValueError("RUNWAYML_API_SECRET environment variable is not set.")
            self.client = RunwayML()
        else:
            self.client = None

    def make_api_call(self, prompt: str, img: np.ndarray, idx:int=None) -> str:
        """
        Makes an API call to generate a video from the provided image and prompt.

        Args:
            prompt (str): The text prompt for generating the video.
            img (np.ndarray): The image to use as the prompt.

        Returns:
            str: The path to the generated video.
        """
        if isinstance(img, np.ndarray):
            pil_image = Image.fromarray(img.astype('uint8'))
        elif isinstance(img, Image.Image):
            pil_image = img
        else:
            raise ValueError("img must be a numpy array or a PIL image")
    
        if self.api == "runway":
            name = "tmp.jpg"
            pil_image.save(name)

            with open(name, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")

            task = self.client.image_to_video.create(
                model='gen3a_turbo',
                prompt_image=f"data:image/png;base64,{base64_image}",
                prompt_text=prompt,
                duration=5
            )
            task_id = task.id

            time.sleep(self.poll_rate) 
            task = self.client.tasks.retrieve(task_id)
            while task.status not in ['SUCCEEDED', 'FAILED']:
                print("polling")
                time.sleep(self.poll_rate) 
                task = self.client.tasks.retrieve(task_id)

            if idx is not None:
                name = f"{idx}"
            else:
                name = prompt[:10].replace('"', '')
            path = f"out_vids/{name}_img2video.mp4"

            urllib.request.urlretrieve(task.output[0], path)

            return path
        else:
            raise ValueError("video api not specified")