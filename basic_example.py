import gradio as gr
import random
import time

import torch
from diffusers import FluxPipeline,FluxTransformer2DModel
import matplotlib.pyplot as plt
import random

from torchao.quantization import quantize_, int8_weight_only
from sd_embed.embedding_funcs import get_weighted_text_embeddings_flux1

import os
import base64
from runwayml import RunwayML
from google.colab import userdata
import urllib.request
from PIL import Image

from utils import *
from agents import *

os.environ["RUNWAYML_API_SECRET"]= userdata.get('runway')

model_id = "black-forest-labs/FLUX.1-dev"

client = RunwayML()

pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.load_lora_weights("/content/simulated_agents/cmbnd2.safetensors")
pipe.to("cuda")

transformer = FluxTransformer2DModel.from_pretrained(
    model_id,
    subfolder = "transformer",
    torch_dtype = torch.bfloat16
)
quantize_(transformer, int8_weight_only())

story_beats = 6

history = []

bob = SyntheticAgent("config_files/bob.yaml")
base_observations = [
    "bob just discovered that alex drank some of his beer.",
    "he clearly wrote a note on it saying 'for bob only!!!!'",
    "he wants to confront alex about it"
]
bob.load_observations(base_observations)

alex = SyntheticAgent("config_files/alex.yaml")
base_observations = [
    "alex is sitting in the living room, drinking one of bob's beers",
    "he knows that he is not supposed to, but he doesn't care",
    "he is looking forward to getting into an argument with bob, he likes riling him up."
]
alex.load_observations(base_observations)

overseer = SyntheticAgent("overseer.yaml")
base_observations = [
    "the overseer is ready to direct the scene",
    "it wants to cultivate a dramatic scene",
    "the scene should end up in a shouting match or a fight"
]
overseer.load_observations(base_observations)

prompt_agent = BaseAgent("prompt.yaml")

character_agents = [alex, bob, overseer]

all_scenes = []

observation = "bob walks into the living room and says ' I thought I told you not to drink my beer!'"
all_scenes.append(observation)

current_beat = 0

interactive = False

def generate_text_for_beat(beat_number):
    global all_scenes
    character_num = len(character_agents)
    concatenated_actions = " ".join(all_scenes[beat_number:beat_number+character_num])
    prompt = prompt_agent.basic_api_call(concatenated_actions)
    for agent in character_agents:
        if agent.name in prompt.lower():
            prompt += agent.flux_caption
    return prompt

def get_default_text(beat_number):
    return generate_text_for_beat(beat_number)

def update_textboxes():
    return [get_default_text(i) for i in range(story_beats)]

def generate_image(prompt):
  num_steps = 50
  width = 1024
  height = 576

  print(prompt)
  prompt_embeds, pooled_prompt_embeds = get_weighted_text_embeddings_flux1(
    pipe= pipe,
    prompt=prompt
  )
  seed = random.randint(0,100000)
  image = pipe(
      # prompt=prompt,
      prompt_embeds=prompt_embeds,
      pooled_prompt_embeds=pooled_prompt_embeds,
      num_inference_steps=num_steps,
      generator=torch.Generator("cuda").manual_seed(seed),
      width=width,
      height=height
  ).images[0]
  return image

def generate_video(prompt,image):
 
    pil_image = Image.fromarray(image.astype('uint8'))

    name = "tmp.jpg"
    pil_image.save(name)

    # encode image to base64
    with open(name, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    # Create a new image-to-video task using the "gen3a_turbo" model
    task = client.image_to_video.create(
      model='gen3a_turbo',
      # Point this at your own image file
      prompt_image=f"data:image/png;base64,{base64_image}",
      prompt_text=prompt,
      duration=5
    )
    task_id = task.id

    # Poll the task until it's complete
    time.sleep(10)  # Wait for a second before polling
    task = client.tasks.retrieve(task_id)
    while task.status not in ['SUCCEEDED', 'FAILED']:
      time.sleep(10)  # Wait for ten seconds before polling
      task = client.tasks.retrieve(task_id)
      print("waiting")

    print('Task complete:', task)
    
    name = prompt[:10].replace('"', '')
    path = f"out_vids/{name}_img2video.mp4"

    urllib.request.urlretrieve(task.output[0], path)

    print(path)

    return path


with gr.Blocks() as demo:
    with gr.Tab("Simulation"):
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox()

        def user(user_message, history: list):
            history.append({"role": "user", "content": user_message})
            return "", history

        def run_agents(history: list):
            global current_beat
            global story_beats
            global all_scenes
            global observation
            global interactive
            this_scene = []

            message = history[-1]["content"]
            if message == "interactive":
                interactive = True

            if interactive:
                if current_beat < story_beats:
                    if message != "":
                        observation = message
                        all_scenes.append(observation)
                        this_scene.append(observation)

                    for agent in character_agents:
                        reflection = agent.reflect()
                        thought = f"<br>{agent.name} thinks: {reflection}</br>"
                        history.append({"role": "assistant", "content": thought})
                        this_scene.append(reflection)
                        observation = agent.process_observation(observation, all_scenes, story_beats, current_beat)
                        all_scenes.append(observation)
                        this_scene.append(observation)

                    this_scene_actions = "\n ".join(this_scene)

                    current_beat += 1

                    history.append({"role": "assistant", "content": this_scene_actions + "\n\n would you like to interact?"})

                    yield history
                else:
                    history.append({"role": "assistant", "content": "The scene is complete"})
                    yield history
            else:
                for i in range(story_beats):
                    for agent in character_agents:
                        time.sleep(3)
                        reflection = agent.reflect()
                        thought = f"<b>{agent.name} thinks: {reflection}</b>"
                        history.append({"role": "assistant", "content": thought})
                        observation = agent.process_observation(observation, all_scenes, story_beats, i)
                        all_scenes.append(observation)
                        history.append({"role": "assistant", "content": observation})
                        yield history

                history.append({"role": "assistant", "content": "The scene is complete"})
                yield history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            run_agents, chatbot, chatbot
        )

    with gr.Tab("Generation"):
        textboxes = []
        update_button = gr.Button("Update Textboxes")
        update_button.click(update_textboxes, outputs=textboxes)

        for i in range(0, story_beats, 3):
            with gr.Row():
                for j in range(3):
                    if i + j < story_beats:
                        with gr.Column():
                            image = gr.Image(label=f"Image for Story Beat {i + j + 1}")
                            default_text = get_default_text(i + j)
                            textbox = gr.Textbox(label=f"Prompt for Story Beat {i + j + 1}", value=default_text)
                            textboxes.append(textbox)
                            image_gen_button = gr.Button(f"Generate for Image {i + j + 1}",
                                      variant="primary")
                            image_gen_button.click(generate_image, inputs=textbox, outputs=image)
                            video_gen_button = gr.Button(f"Generate for Video {i + j + 1}",
                                      variant="primary")
                            video = gr.Video(label=f"Video for Story Beat {i + j + 1}")
                            video_gen_button.click(generate_video, inputs=[textbox,image], outputs=video)
                            
                            
        

demo.launch(debug=True)