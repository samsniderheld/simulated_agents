import numpy as np
import random
import torch

from diffusers import FluxPipeline, FluxImg2ImgPipeline, FluxTransformer2DModel
from PIL import Image
from torchao.quantization import quantize_, int8_weight_only
from sd_embed.embedding_funcs import get_weighted_text_embeddings_flux1

class FluxWrapper:
    """
    A wrapper class for the FluxPipeline and FluxTransformer2DModel to generate images using a pre-trained model and LoRA weights.

    Attributes:
        model_id (str): The ID of the pre-trained model.
        lora_path (str): The path to the LoRA weights.
        pipe (FluxPipeline): The FluxPipeline object.
        transformer (FluxTransformer2DModel): The FluxTransformer2DModel object.
        img2img (bool): Are we using the img2img process?
    """

    def __init__(self, model_id: str, lora_paths: [str], img2img:bool=True) -> None:
        """
        Initializes the FluxWrapper with the specified model ID and LoRA weights path.

        Args:
            model_id (str): The ID of the pre-trained model.
            lora_path (str): The path to the LoRA weights.
        """
        self.model_id = model_id
        self.lora_paths = lora_paths
        self.pipe = None
        self.transformer = None
        self.img2img = img2img
        self.load_model()

    def load_model(self) -> None:
        """
        Loads the pre-trained model and LoRA weights, and quantizes the transformer model.
        """

        if self.img2img:
            self.pipe = FluxImg2ImgPipeline.from_pretrained(self.model_id, torch_dtype=torch.bfloat16)
            for i,path in enumerate(self.lora_paths):
                self.pipe.load_lora_weights(path,adapter_name=f"lora_{i}")
            self.pipe.to("cuda")
        else:
            self.pipe = FluxPipeline.from_pretrained(self.model_id, torch_dtype=torch.bfloat16)
            for i,path in enumerate(self.lora_paths):
                self.pipe.load_lora_weights(path,adapter_name=f"lora_{i}")
            self.pipe.to("cuda")
        self.transformer = FluxTransformer2DModel.from_pretrained(
            self.model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16
        )
        quantize_(self.transformer, int8_weight_only())

    def generate_image(self, prompt: str, seed: int = None, steps: int = 40, lora_0_weight: float = 0.3 ,lora_1_weight: float = 1.0, width: int = 1024, height: int = 576,img2img_str=0.7) -> torch.Tensor:
        """
        Generates an image based on the provided prompt.

        Args:
            prompt (str): The text prompt for generating the image.
            seed (int, optional): The random seed for generating the image. Defaults to None.
            width (int, optional): The width of the generated image. Defaults to 1024.
            height (int, optional): The height of the generated image. Defaults to 576.
            steps (int, optional): The number of inference steps. Defaults to 40.

        Returns:
            torch.Tensor: The generated image.
        """
        prompt_embeds, pooled_prompt_embeds = get_weighted_text_embeddings_flux1(
            pipe=self.pipe,
            prompt=prompt
        )
        if seed is None:
            seed = random.randint(0, 100000)

        self.pipe.set_adapters(["lora_0", "lora_1"], adapter_weights=[lora_0_weight, lora_1_weight])
        
        if self.img2img:

            np_img = np.random.rand(width,height,3) * 255
            noise_image = Image.fromarray(np_img.astype('uint8'))

            # first generation, no img2img
            image = self.pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=steps,
                generator=torch.Generator("cuda").manual_seed(seed),
                width=width,
                height=height,
                strength=1.0,
                image=noise_image
            ).images[0]

            image = self.pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=steps,
                generator=torch.Generator("cuda").manual_seed(seed),
                width=width,
                height=height,
                strength=img2img_str,
                image=image
            ).images[0]
            
        else:
            image = self.pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=steps,
                generator=torch.Generator("cuda").manual_seed(seed),
                width=width,
                height=height
            ).images[0]

        return image