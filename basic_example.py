import gradio as gr
import random
import time

from utils import *
from agents import *

story_beats = 3

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
    return prompt

def get_default_text(beat_number):
    return generate_text_for_beat(beat_number)

def update_textboxes():
    return [get_default_text(i) for i in range(story_beats)]

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
        for i in range(0, story_beats, 3):
            with gr.Row():
                for j in range(3):
                    if i + j < story_beats:
                        with gr.Column():
                            gr.Image(label=f"Image for Story Beat {i + j + 1}")
                            default_text = get_default_text(i + j)
                            textbox = gr.Textbox(label=f"Text for Story Beat {i + j + 1}", value=default_text)
                            textboxes.append(textbox)
                            gr.Button(f"Generate for Story Beat {i + j + 1}", 
                                      variant="primary")
        update_button = gr.Button("Update Textboxes")
        update_button.click(update_textboxes, outputs=textboxes)

demo.launch(debug=True)