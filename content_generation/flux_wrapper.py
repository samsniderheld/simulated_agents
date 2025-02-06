import random
import torch

from diffusers import FluxPipeline,FluxTransformer2DModel
from torchao.quantization import quantize_, int8_weight_only
from sd_embed.embedding_funcs import get_weighted_text_embeddings_flux1

class FluxWrapper():
    def __init__(self, model_id, lora_path):
        self.model_id = model_id
        self.lora_path = lora_path
        self.pipe = None
        self.transformer = None
        self.load_model()

    def load_model(self):
        self.pipe = FluxPipeline.from_pretrained(self.model_id, torch_dtype=torch.bfloat16)
        self.pipe.load_lora_weights(self.lora_path)
        self.pipe.to("cuda")

        self.transformer = FluxTransformer2DModel.from_pretrained(
            self.model_id,
            subfolder = "transformer",
            torch_dtype = torch.bfloat16
        )
        quantize_(self.transformer, int8_weight_only())

    def generate_image(self, prompt, seed, width=1024, height=576,steps=40):
        prompt_embeds, pooled_prompt_embeds = get_weighted_text_embeddings_flux1(
            pipe= self.pipe,
            prompt=prompt
        )
        if seed is None:
            seed = random.randint(0,100000)
        image = self.pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=steps,
            generator=torch.Generator("cuda").manual_seed(seed),
            width=width,
            height=height
        ).images[0]
        return image