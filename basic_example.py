import gradio as gr
import os
import time

from utils import *
from agents import *
from content_generation import *

os.makedirs('out_imgs', exist_ok=True)
os.makedirs('out_vids', exist_ok=True)

print("loading agents")
story_beats = 3

history = []

synthetic_agents, helper_agents = instantiate_agents("config_files/scenario.yaml")

alex = get_agent_by_name("alex", synthetic_agents)
bob = get_agent_by_name("bob", synthetic_agents)
director = get_agent_by_name("director", synthetic_agents)

prompt_agent = get_agent_by_name("prompt", helper_agents)

character_agents = [alex, bob, director]

all_scenes = []

observation = "bob walks into the living room and says ' I thought I told you not to drink my beer!'"
all_scenes.append(observation)

current_beat = 0

interactive = False


print("loading content generation capabilities")
image_gen = FluxWrapper("black-forest-labs/FLUX.1-dev", "lora/cmbnd2.safetensors")
video_gen = VideoWrapper(api="runway")
print("loading complete")

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
    text_responses = []
    for i in range(story_beats):
        default_text = get_default_text(i)
        text_responses.append(default_text)
        text_responses.append(default_text)  # Duplicate the text response
    return text_responses

def generate_image(prompt):
    image = image_gen.generate_image(prompt)
    return image

def generate_video(prompt,image):
    path = video_gen.make_api_call(prompt,image)    
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
                        thought = f"<b style='color:green;'>{agent.name} thinks: {reflection}</b>"
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
                        thought = f"<b style='color:green;'>{agent.name} thinks: {reflection}</b>"
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
                            with gr.Tab("image"):
                              image = gr.Image(label=f"Image for Story Beat {i + j + 1}")
                              default_text = get_default_text(i + j)
                              textbox = gr.Textbox(label=f"Prompt for Story Beat {i + j + 1}", value=default_text)
                              textboxes.append(textbox)
                              image_gen_button = gr.Button(f"Generate for Image {i + j + 1}",
                                        variant="primary")
                              image_gen_button.click(generate_image, inputs=textbox, outputs=image)
                            with gr.Tab("video"):
                              video = gr.Video(label=f"Video for Story Beat {i + j + 1}")
                              textbox_2 = gr.Textbox(label=f"Prompt for Story Beat {i + j + 1}", value=default_text)
                              textboxes.append(textbox_2)
                              video_gen_button = gr.Button(f"Generate for Video {i + j + 1}",
                                        variant="primary")
                              video_gen_button.click(generate_video, inputs=[textbox_2,image], outputs=video)
                            
                            
demo.launch(debug=True)