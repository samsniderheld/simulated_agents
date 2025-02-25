import base64
import numpy as np
import jwt
import os
import requests
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

    def __init__(self, api: str = "kling", poll_rate: int = 10) -> None:
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

        elif self.api == "kling":
            self.api_key = os.getenv("Kling_API_KEY")
            if not self.api_key:
                raise ValueError("Kling_API_KEY environment variable is not set.")
            self.secret_key = os.getenv("Kling_API_SECRET")  
            if not self.secret_key:
                raise ValueError("Kling_API_SECRET environment variable is not set.")
        else:
            self.client = None

    def encode_jwt_token(self, access_key, secret_key):
        headers = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "iss": access_key,
            "exp": int(time.time()) + 1800,
            "nbf": int(time.time()) - 5
        }
        return jwt.encode(payload, secret_key, algorithm="HS256")
    

    def make_api_call(self, prompt: str, img: np.ndarray, idx:int=None, duration:int=5) -> str:
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
    
        name = "tmp.jpg"
        pil_image.save(name)

        with open(name, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        if self.api == "runway":
            
            task = self.client.image_to_video.create(
                model='gen3a_turbo',
                prompt_image=f"data:image/png;base64,{base64_image}",
                prompt_text=prompt,
                duration=duration
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

            try:
                urllib.request.urlretrieve(task.output[0], path)
            except:
                print("error downloading video")
                print(task)
                return None
            
            print(f"video saved to {path}")

            return path
        
        elif self.api == "kling":

            api_token = self.encode_jwt_token(self.api_key,self.secret_key)
            
            headers = {
                'content-type': 'application/json;charset=utf-8',
                'Authorization': f'Bearer {api_token}'
            }

            data = {
                "model": "kling-v1-6",
                "image": base64_image,
                "prompt": prompt,
                "negative_prompt": "poor quality",
                "cfg_scale": 0.5,
                "mode": "std",
                "duration": duration
            }

            video_urls = []

            response = requests.post(submit_url, headers=headers, json=data)
            response_json = response.json()
            if response_json.get("code") != 0:
                print(f"Request error: {response.text}")
                return []
            
            task_id = response_json["data"]["task_id"]

            time.sleep(self.poll_rate) 
            result_url = f"https://api.klingai.com/v1/videos/image2video/{task_id}"
            response_task = requests.get(result_url, headers=headers)
            response_json_task = response_task.json()

            while response_json_task['data']['task_status'] not in ['succeed', 'failed']:
                print("polling")
                if response_json_task.get("code") != 0:
                    print(f"Task {task_id} error: {response_task.text}")
                    break

                time.sleep(self.poll_rate) 
                result_url = f"https://api.klingai.com/v1/videos/image2video/{task_id}"
                response_task = requests.get(result_url, headers=headers)
                response_json_task = response_task.json()

            
            if 'failed' in response_json_task['data']['task_status']:
                print("video generation failed")
                print(response_json_task)
                
            videos_result_list = response_json_task['data']['task_result']['videos']
            video_urls = [video['url'] for video in videos_result_list]

            if idx is not None:
                name = f"{idx}"
            else:
                name = prompt[:10].replace('"', '')

            path = f"out_vids/{name}_img2video.mp4"

            try:
                urllib.request.urlretrieve(video_urls[0], path)
            except:
                print("error downloading video")
                print(video_urls[0])
                return None
            
            print(f"video saved to {path}")

            return path
        else:
            raise ValueError("video api not specified")