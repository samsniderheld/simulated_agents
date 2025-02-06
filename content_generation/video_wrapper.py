import base64
import numpy as np
import os
import time
import urllib.request

from PIL import Image
from runwayml import RunwayML

class VideoWrapper:
    def __init__(self, api: str = "runway", poll_rate: int=10) -> None:
        self.api = api
        self.poll_rate = poll_rate 
        if self.api == "runway":
            api_key = os.getenv("RUNWAYML_API_SECRET")
            if not api_key:
                raise ValueError("RUNWAYML_API_SECRET environment variable is not set.")
            self.client = RunwayML()
        else:
            self.client = None

    def make_api_call(self, prompt: str, img: np.ndarray) -> str:
        if self.api == "runway":
            pil_image = Image.fromarray(img.astype('uint8'))

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

            
            name = prompt[:10].replace('"', '')
            path = f"out_vids/{name}_img2video.mp4"

            urllib.request.urlretrieve(task.output[0], path)

            return path
        else:
            raise ValueError("video api not specified")
