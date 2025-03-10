import argparse
import os
import shutil

import gradio as gr

from functools import partial

from utils import *
from agents import *
from content_generation import *


parser = argparse.ArgumentParser(description="Simulated Agents")
parser.add_argument('--iterations', type=int, default=3, help='Number of story beats')
parser.add_argument('--scenario_file_path', type=str, default='config_files/scenario.yaml', help='Path to the scenario file')
parser.add_argument('--share', action='store_true', help='Share the Gradio app')
parser.add_argument('--debug', action='store_true', help='Debug mode')
parser.add_argument('--interactive', action='store_true', help='Interactive mode')
parser.add_argument('--show_simulated_thinking', action='store_true', help='show the simulated thinking')
parser.add_argument('--narrative', type=str, default='today we are writing a story about a young boy learning about the universe', help='what are we writing an episode about')

args = parser.parse_args()

os.makedirs('out_imgs', exist_ok=True)
os.makedirs('out_vids', exist_ok=True)
os.makedirs('out_audio', exist_ok=True)
os.makedirs('final_vids', exist_ok=True)

# Create and clear necessary directories
directories = ['out_imgs', 'out_vids', 'out_audio', 'combined_assets', 'final_vids']
for directory in directories:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

iterations = args.iterations

history = []

all_scenes = []

current_iteration = 0

interactive = args.interactive

show_simulated_thinking = args.show_simulated_thinking

print("loading agents")
synthetic_agents, helper_agents = instantiate_agents(args.scenario_file_path)
script_writer = get_agent_by_name("script_writer", synthetic_agents)
producer = get_agent_by_name("producer", synthetic_agents)
img_prompt_agent = get_agent_by_name("img_prompt", helper_agents)
vid_prompt_agent = get_agent_by_name("vid_prompt", helper_agents)

character_agents = [script_writer, producer]

observation = args.narrative
all_scenes.append(observation)

print("loading content generation capabilities")
image_gen = FluxWrapper("black-forest-labs/FLUX.1-dev", ["lora/ARCANE_STYLE_FADOO-FLUX.safetensors", "lora/taylrrdect-v5.safetensors"])
video_gen = VideoWrapper(api="kling")
tts = TTSWrapper(api="eleven_labs")
print("loading complete")

final_script = None

def update_textboxes():
    """
    A function to fill the generation tabs inputs with the generated content for each shot.

    Returns:
        list: A list of text responses for each shot.
    """
    global final_script
    text_responses = []
    num_shots = len(final_script.shots)
    for i in range(num_shots):
        action_text = final_script.shots[i].shot_action
        img_text = f"{script_writer.lora_key_word}, {final_script.shots[i].txt2img_prompt}, {script_writer.flux_caption}" 
        vid_text = img_text
        vo_text = final_script.shots[i].vo
        text_responses.append(action_text)
        text_responses.append(img_text)
        text_responses.append(vid_text)
        text_responses.append(vo_text)
    return text_responses

def augment_text_prompt(prompt):
    """
    Augments the given prompt using the image prompt agent.

    Args:
        prompt (str): The prompt to augment.

    Returns:
        str: The augmented prompt.
    """
    augmented_prompt = img_prompt_agent.basic_api_call(prompt)
    output = f"{script_writer.lora_key_word},\n\n{augmented_prompt}\n\n Costume: {script_writer.flux_caption}" 

    return output

def augment_video_prompt(prompt):
    """
    Augments the given prompt using the video prompt agent.

    Args:
        prompt (str): The prompt to augment.

    Returns:
        str: The augmented prompt.
    """
    augmented_prompt = vid_prompt_agent.basic_api_call(prompt)

    return augmented_prompt

def create_video():
    """
    Creates a final video by combining video and audio files.

    Returns:
        str: The path to the final video.
    """
    video_path = "out_vids"
    audio_path = "out_audio"
    out_path = "final_vids/final_video.mp4"

    video_files = sorted(os.listdir(video_path))
    audio_files = sorted(os.listdir(audio_path))

    clips = []

    for video_file, audio_file in zip(video_files, audio_files):
        video_file_path = os.path.join(video_path, video_file)
        audio_file_path = os.path.join(audio_path, audio_file)
        combine_video_audio(video_file_path, audio_file_path, f"combined_assets/{video_file}")
        clips.append(f"combined_assets/{video_file}")

    concatenate_videos(clips, out_path)

    return out_path

def user(user_message, history: list):
    """
    Handles user input and updates the chat history.

    Args:
        user_message (str): The user's message.
        history (list): The chat history.

    Returns:
        tuple: An empty string and the updated chat history.
    """
    history.append({"role": "user", "content": user_message})
    return "", history

def run_agents(history: list):
    """
    Runs the agents to process the scene and generate content. This is the main script generation loop.

    Args:
        history (list): The chat history.

    Yields:
        list: The updated chat history.
    """
    global current_iteration
    global iterations
    global all_scenes
    global observation
    global interactive
    global final_script

    message = history[-1]["content"]

    if interactive:
        iteration = 0
        if current_iteration < iterations:
            if message != "":
                observation = message
                all_scenes.append(observation)
            
            if iteration == 0:
                script = script_writer.process_observation(observation, all_scenes, use_structured=True)
            else:
                script = script_writer.process_observation(f"please make the following changes to the orignal script: {observation}", all_scenes, use_structured=True)
            
            script_str = script.to_str()
            print(script_str)
            all_scenes.append(script_str)
            new_script = f"<b style='color:green;'>{script_writer.name}: \n\n {script_str}</b>"
            history.append({"role": "assistant", "content": new_script})

            iteration += 1
            history.append({"role": "assistant", "content": "<b style='color:white'>: Would do you think?</b>"})

            yield history
        else:
            history.append({"role": "assistant", "content": "The scene is complete"})
            yield history

    else:
        # Simulate the scene
        print("simulating scene")
        for i in range(iterations):
                
            for agent in character_agents:
                if agent.name == "script_writer":
                    if i == 0:
                        script = script_writer.process_observation(observation, all_scenes, use_structured=True)
                    else:
                        script = script_writer.process_observation(f"please make the following changes to the orignal script: {observation}", all_scenes, use_structured=True)
                    
                    script_str = script.to_str()
                    print(script_str)
                    all_scenes.append(script_str)
                    new_script = f"<b style='color:green;'>{agent.name}: \n\n {script_str}</b>"
                    history.append({"role": "assistant", "content": new_script})
                    
                    yield history

                elif agent.name == "producer":

                    observation = producer.process_observation(f"what do you think of : {script_str} tell the script writer what they should change", all_scenes)
                    print(observation)
                    all_scenes.append(observation)
                    critique = f"<b style='color:white;'>{agent.name} thinks: \n\n {observation}</b>"
                    history.append({"role": "assistant", "content": critique})
                    
                    yield history

        final_script = script_writer.process_observation(f"please make the following changes to the orignal script: {observation}", all_scenes, use_structured=True)
        final_script_str = final_script.to_str()
        new_script = f"<b style='color:green;'>script writer: \n\n {final_script_str}</b>"
        history.append({"role": "assistant", "content": new_script})

        history.append({"role": "assistant", "content": "The scene is complete"})
        yield history


# below the gradio interface is defined
with gr.Blocks() as demo:
    with gr.Tab("Simulation"):
        chatbot = gr.Chatbot(type="messages")
        
        if interactive:
            msg = gr.Textbox()
            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                run_agents, chatbot, chatbot
            )
        else:
            msg = gr.Button("run simulation")
            msg.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                run_agents, chatbot, chatbot
            )

    with gr.Tab("Generation"):
        text_boxes = []
        update_button = gr.Button("Update Textboxes")
        update_button.click(update_textboxes, outputs=text_boxes)
        for i in range(0, 6, 3):
            with gr.Row():
                for j in range(3):
                    if i + j < 6:
                        with gr.Column():
                            with gr.Tab("image"):
                              image = gr.Image(label=f"Image for Story Beat {i + j}")
                              action_box = gr.Textbox(label=f"scene {i + j}", value="")
                              textbox = gr.Textbox(label=f"Prompt for Story Beat {i + j}", value="")
                              augment_img_prompt_button = gr.Button("Augment Prompt", variant="primary")
                              augment_img_prompt_button.click(augment_text_prompt, inputs=textbox, outputs=textbox)
                              seed = gr.Number(label="seed", value=0)
                              steps = gr.Number(label="steps", value=40)
                              lora_0_weight = gr.Slider(label="lora_0_weight", minimum=0.0, maximum=1.0, value=0.5, step=0.1)
                              lora_1_weight = gr.Slider(label="lora_1_weight", minimum=0.0, maximum=1.0, value=1.0, step=0.1)
                              image_gen_button = gr.Button(f"Generate Image {i + j}",
                                        variant="primary")
                              image_gen_button.click(image_gen.generate_image, inputs=[textbox,seed,steps,lora_0_weight,lora_1_weight], outputs=image)
                              text_boxes.append(action_box)
                              text_boxes.append(textbox)

                            with gr.Tab("video"):
                              video = gr.Video(label=f"Video for Story Beat {i + j}")
                              textbox_2 = gr.Textbox(label=f"Prompt for Story Beat {i + j}", value="")
                              augment_vid_prompt_button = gr.Button("Augment Prompt", variant="primary")
                              duration = gr.Dropdown(label="duration", choices=[5, 10], value=5)
                              augment_vid_prompt_button.click(augment_video_prompt, inputs=textbox_2, outputs=textbox_2)
                              video_gen_button = gr.Button(f"Generate Video {i + j}",
                                        variant="primary")
                              video_gen_button.click(partial(video_gen.make_api_call, idx=i + j), inputs=[textbox_2,image,duration], outputs=video)
                              textbox_3 = gr.Textbox(label=f"Prompt for VO {i + j}", value="")
                              tts_seed = gr.Number(label="seed", value=0)
                              audio = gr.Audio()
                              audio_gen_button = gr.Button(f"Generate Audio {i + j}",
                                        variant="primary")
                              audio_gen_button.click(partial(tts.make_api_call, idx=i + j), inputs=[textbox_3,tts_seed], outputs=audio)
                              text_boxes.append(textbox_2)
                              text_boxes.append(textbox_3)
    with gr.Tab("output"): 
        final_video = gr.Video(label=f"final video")                      
        gr.Button("create_video").click(create_video, outputs=final_video)

demo.launch(debug=args.debug, share=args.share, server_port=9000)