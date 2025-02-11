import argparse
import os
import shutil
import time

import gradio as gr

from functools import partial

from utils import *
from agents import *
from content_generation import *

from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips


parser = argparse.ArgumentParser(description="Simulated Agents")
parser.add_argument('--story_beats', type=int, default=3, help='Number of story beats')
parser.add_argument('--scenario_file_path', type=str, default='config_files/scenario.yaml', help='Path to the scenario file')
parser.add_argument('--share', action='store_true', help='Share the Gradio app')
parser.add_argument('--debug', action='store_true', help='Debug mode')
parser.add_argument('--interactive', action='store_true', help='Interactive mode')
parser.add_argument('--show_simulated_thinking', action='store_true', help='show the simulated thinking')

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

story_beats = args.story_beats

history = []

all_scenes = []

current_beat = 0

interactive = args.interactive

show_simulated_thinking = args.show_simulated_thinking

print("loading agents")
synthetic_agents, helper_agents = instantiate_agents(args.scenario_file_path)
script_writer = get_agent_by_name("script_writer", synthetic_agents)
producer = get_agent_by_name("producer", synthetic_agents)

img_prompt_agent = get_agent_by_name("img_prompt", helper_agents)
vid_prompt_agent = get_agent_by_name("vid_prompt", helper_agents)
audio_prompt_agent = get_agent_by_name("audio_prompt", helper_agents)


character_agents = [script_writer, producer]

observation = "today we are writing a story about a young boy learning about the universe"
all_scenes.append(observation)

print("loading content generation capabilities")
image_gen = FluxWrapper("black-forest-labs/FLUX.1-dev", "lora/Realistic_PixArt_Doodle_art_style.safetensors")
video_gen = VideoWrapper(api="runway")
tts = TTSWrapper(api="eleven_labs")
print("loading complete")

final_script = None

def concatenate_scenes(beat_number):
    global all_scenes
    character_num = len(character_agents)
    start = int(beat_number*character_num)
    end = start + character_num
    concatenated_actions = " ".join(all_scenes[start:end])
    return concatenated_actions

def generate_img_text_for_beat(beat_number):
    global final_script
    prompt = final_script.shots[beat_number].txt2img_prompt
    return f"chibi, {prompt}, super hyper realistic, cinematic volumetric lighting, shot with Sony Fx6"

    # scene = concatenate_scenes(beat_number)
    # prompt = img_prompt_agent.basic_api_call(scene)
    # for agent in character_agents:
    #     if agent.name in prompt.lower():
    #         prompt += f"\n\n{agent.flux_caption}"
    # return prompt

def generate_vid_text_for_beat(img_text):
    prompt = vid_prompt_agent.basic_api_call(img_text)
    return prompt

def generate_audio_text_for_beat(beat_number):
    global final_script
    prompt = final_script.shots[beat_number].vo
    return prompt
    
    # scene = concatenate_scenes(beat_number)
    # prompt = audio_prompt_agent.basic_api_call(scene)
    # return prompt

def update_textboxes():
    text_responses = []
    for i in range(story_beats):
        img_text = generate_img_text_for_beat(i)
        text_responses.append(img_text)
        vid_text = generate_vid_text_for_beat(img_text)
        text_responses.append(vid_text) 
        audio_text = generate_audio_text_for_beat(i)
        text_responses.append(audio_text)  
    return text_responses

def generate_image(prompt):
    image = image_gen.generate_image(prompt)
    return image

def generate_video(prompt,image,idx):
    path = video_gen.make_api_call(prompt,image,idx)    
    return path

def generate_audio(prompt,idx):
    path = tts.make_api_call(prompt,idx)
    return path

def combine_video_audio(video_path, audio_path, output_path):
    """
    Combines video and audio into a single file.

    Args:
        video_path (str): The path to the video file.
        audio_path (str): The path to the audio file.
        output_path (str): The path to the output file.
    """
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    final_clip = video_clip.with_audio(audio_clip)
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

def concatenate_videos(video_paths, output_path):
    """
    Concatenates multiple video files into a single file.

    Args:
        video_paths (list): List of paths to the video files.
        output_path (str): The path to the output file.
    """
    clips = [VideoFileClip(video) for video in video_paths]
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

def create_video():
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
        history.append({"role": "user", "content": user_message})
        return "", history

def run_agents(history: list):
    global current_beat
    global story_beats
    global all_scenes
    global observation
    global interactive
    global final_script
    this_scene = []

    message = history[-1]["content"]

    # Simulate the scene
    print("simulating scene")
    for i in range(story_beats):
        for agent in character_agents:
            if agent.name == "script_writer":
                if i == 0:
                    script = script_writer.process_observation(observation, all_scenes, story_beats, i, use_json=True)
                else:
                    script = script_writer.process_observation(f"please make the following changes to the orignal script: {observation}", all_scenes, story_beats, i, use_json=True)
                
                script_str = script.to_str()
                print(script_str)
                all_scenes.append(script_str)
                new_script = f"<b style='color:green;'>{agent.name}: \n\n {script_str}</b>"
                history.append({"role": "assistant", "content": new_script})
                
                yield history

            elif agent.name == "producer":

                observation = producer.process_observation(f"what do you think of : {script_str} tell the script writer what they should change", all_scenes, story_beats, i)
                print(observation)
                all_scenes.append(observation)
                critique = f"<b style='color:white;'>{agent.name} thinks: \n\n {observation}</b>"
                history.append({"role": "assistant", "content": critique})
                
                yield history

    final_script = script

    history.append({"role": "assistant", "content": "The scene is complete"})
    yield history

    # if interactive:
    #     if current_beat < story_beats:
    #         if message != "":
    #             observation = message
    #             all_scenes.append(observation)
    #             this_scene.append(observation)

    #         for agent in character_agents:
    #             if show_simulated_thinking:
    #                 reflection = agent.reflect()
    #                 thought = f"<b style='color:green;'>{agent.name} thinks: {reflection}</b>"
    #                 history.append({"role": "assistant", "content": thought})
    #                 this_scene.append(reflection)
    #             observation = agent.process_observation(observation, all_scenes, story_beats, current_beat)
    #             all_scenes.append(observation)
    #             this_scene.append(observation)

    #         this_scene_actions = "\n ".join(this_scene)

    #         current_beat += 1

    #         history.append({"role": "assistant", "content": this_scene_actions + "\n\n would you like to interact?"})

    #         yield history
    #     else:
    #         history.append({"role": "assistant", "content": "The scene is complete"})
    #         yield history
    # else:
    #     for i in range(story_beats):
    #         for agent in character_agents:
    #             time.sleep(3)
    #             if show_simulated_thinking:
    #                 reflection = agent.reflect()
    #                 thought = f"<b style='color:green;'>{agent.name} thinks: {reflection}</b>"
    #                 history.append({"role": "assistant", "content": thought})
    #                 this_scene.append(reflection)
    #             observation = agent.process_observation(observation, all_scenes, story_beats, i)
    #             all_scenes.append(observation)
    #             history.append({"role": "assistant", "content": observation})
    #             yield history

    #     history.append({"role": "assistant", "content": "The scene is complete"})
    #     yield history


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

        for i in range(0, 6, 3):
            with gr.Row():
                for j in range(3):
                    if i + j < 6:
                        with gr.Column():
                            with gr.Tab("image"):
                              image = gr.Image(label=f"Image for Story Beat {i + j}")
                              default_text = "input"

                              img_text_button = gr.Button("Generate Prompt")
                              textbox = gr.Textbox(label=f"Prompt for Story Beat {i + j}", value=default_text)
                              img_text_button.click(partial(generate_img_text_for_beat, i + j), outputs=textbox)

                              image_gen_button = gr.Button(f"Generate for Image {i + j}",
                                        variant="primary")
                              image_gen_button.click(generate_image, inputs=textbox, outputs=image)

                            with gr.Tab("video"):
                              video = gr.Video(label=f"Video for Story Beat {i + j}")
                              
                              vid_text_button = gr.Button("Generate Video Prompt")
                              textbox_2 = gr.Textbox(label=f"Prompt for Story Beat {i + j}", value=default_text)
                              vid_text_button.click(partial(generate_img_text_for_beat, i + j), outputs=textbox_2)

                              video_gen_button = gr.Button(f"Generate for Video {i + j}",
                                        variant="primary")
                              video_gen_button.click(partial(generate_video, idx=i + j), inputs=[textbox_2,image], outputs=video)

                              audio_text_button = gr.Button("Generate Audio Prompt")
                              textbox_3 = gr.Textbox(label=f"Prompt for VO {i + j}", value=default_text)
                              
                              audio_text_button.click(partial(generate_audio_text_for_beat, i + j), outputs=textbox_3)
                              audio = gr.Audio()
                              audio_gen_button = gr.Button(f"Generate for Audio {i + j}",
                                        variant="primary")
                              audio_gen_button.click(partial(generate_audio, idx=i + j), inputs=[textbox_3,], outputs=audio)

    with gr.Tab("output"): 
        final_video = gr.Video(label=f"final video")                      
        gr.Button("create_video").click(create_video, outputs=final_video)

demo.launch(debug=args.debug, share=args.share, server_port=9000)