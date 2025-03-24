import numpy as np
import random
import torch

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image

class SDXLWrapper:
    """
    A wrapper class for the StableDiffusionXLPipeline to generate images using a pre-trained SDXL model and LoRA weights.

    Attributes:
        model_id (str): The ID of the pre-trained model.
        lora_paths (list[str]): The paths to the LoRA weights.
        pipe (StableDiffusionXLPipeline): The SDXL pipeline object.
        img2img (bool): Whether to use the img2img process.
    """

    def __init__(self, model_id: str, lora_paths: [str], img2img: bool = True) -> None:
        """
        Initializes the SDXLWrapper with the specified model ID and LoRA weights paths.

        Args:
            model_id (str): The ID of the pre-trained SDXL model.
            lora_paths (list[str]): The paths to the LoRA weights.
            img2img (bool): Whether to use the img2img process. Defaults to True.
        """
        self.model_id = model_id
        self.lora_paths = lora_paths
        self.pipe = None
        self.img2img = img2img
        self.load_model()

    def load_model(self) -> None:
        """
        Loads the pre-trained SDXL model and LoRA weights.
        """
        if self.img2img:
            self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16)
        else:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16)

        # Load LoRA weights
        for i, path in enumerate(self.lora_paths):
            self.pipe.load_lora_weights(path, adapter_name=f"lora_{i}")

        self.pipe.to("cuda")

    def generate_image(self, prompt: str, seed: int = None, steps: int = 40, lora_weights: list[float] = None, width: int = 1024, height: int = 576, img2img_strength: float = 0.7) -> Image.Image:
        """
        Generates an image based on the provided prompt.

        Args:
            prompt (str): The text prompt for generating the image.
            seed (int, optional): The random seed for generating the image. Defaults to None.
            steps (int, optional): The number of inference steps. Defaults to 40.
            lora_weights (list[float], optional): The weights for the LoRA adapters. Defaults to None.
            width (int, optional): The width of the generated image. Defaults to 1024.
            height (int, optional): The height of the generated image. Defaults to 576.
            img2img_strength (float, optional): The strength for the img2img process. Defaults to 0.7.

        Returns:
            Image.Image: The generated image.
        """
        if lora_weights is None:
            lora_weights = [1.0] * len(self.lora_paths)

        # Set LoRA adapter weights
        adapter_names = [f"lora_{i}" for i in range(len(self.lora_paths))]
        self.pipe.set_adapters(adapter_names, adapter_weights=lora_weights)

        # Generate random seed if not provided
        if seed is None:
            seed = random.randint(0, 100000)

        generator = torch.Generator("cuda").manual_seed(seed)

        if self.img2img:
            # Create a random noise image
            noise = np.random.rand(height, width, 3) * 255
            noise_image = Image.fromarray(noise.astype('uint8'))

            # Generate image using img2img
            image = self.pipe(
                prompt=prompt,
                image=noise_image,
                num_inference_steps=steps,
                generator=generator,
                strength=img2img_strength
            ).images[0]
        else:
            # Generate image using text-to-image
            image = self.pipe(
                prompt=prompt,
                num_inference_steps=steps,
                generator=generator,
                width=width,
                height=height
            ).images[0]

        return image