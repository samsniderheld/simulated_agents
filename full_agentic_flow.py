import argparse
import os
import shutil

from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips

from utils import *
from agents import *
from content_generation import *

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Simulated Agents")
parser.add_argument('--story_beats', type=int, default=3, help='Number of story beats')
parser.add_argument('--scenario_file_path', type=str, default='config_files/scenario.yaml', help='Path to the scenario file')
parser.add_argument('--variations', type=int, default=1, help='Number of variations')
parser.add_argument('--interactive', action='store_true', help='Interactive mode')
args = parser.parse_args()

def create_directory(directory):
    """
    Makes directory. Clears all files in the specified directory if it exists.

    Args:
        directory (str): The path to the directory to clear.
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

def concatenate_scenes(beat_number, all_scenes):
    """
    Concatenates scenes for a given beat number.

    Args:
        beat_number (int): The beat number.
        all_scenes (list): List of all scenes.

    Returns:
        str: Concatenated actions for the beat number.
    """
    character_num = len(character_agents)
    concatenated_actions = " ".join(all_scenes[beat_number:beat_number+character_num])
    return concatenated_actions

def generate_img_text_for_beat(beat_number, all_scenes):
    """
    Generates image text for a given beat number.

    Args:
        beat_number (int): The beat number.
        all_scenes (list): List of all scenes.

    Returns:
        str: Generated image text.
    """
    scene = concatenate_scenes(beat_number, all_scenes)
    prompt = img_prompt_agent.basic_api_call(scene)
    for agent in character_agents:
        if agent.name in prompt.lower():
            prompt += f"\n\n{agent.flux_caption}"
    return prompt

def generate_vid_text_for_beat(img_text):
    """
    Generates video text for a given image text.

    Args:
        img_text (str): The image text.

    Returns:
        str: Generated video text.
    """
    prompt = vid_prompt_agent.basic_api_call(img_text)
    return prompt

def generate_audio_text_for_beat(beat_number, all_scenes):
    """
    Generates audio text for a given beat number.

    Args:
        beat_number (int): The beat number.
        all_scenes (list): List of all scenes.

    Returns:
        str: Generated audio text.
    """
    scene = concatenate_scenes(beat_number, all_scenes)
    prompt = audio_prompt_agent.basic_api_call(scene)
    return prompt

def generate_image(prompt):
    """
    Generates an image based on the provided prompt.

    Args:
        prompt (str): The text prompt for generating the image.

    Returns:
        Image: The generated image.
    """
    image = image_gen.generate_image(prompt)
    return image

def generate_video(prompt, image):
    """
    Generates a video based on the provided prompt and image.

    Args:
        prompt (str): The text prompt for generating the video.
        image (Image): The image to use as the prompt.

    Returns:
        str: The path to the generated video.
    """
    path = video_gen.make_api_call(prompt, image)    
    return path

def generate_audio(prompt):
    """
    Generates audio based on the provided prompt.

    Args:
        prompt (str): The text prompt for generating the audio.

    Returns:
        str: The path to the generated audio.
    """
    path = tts.make_api_call(prompt)
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

# Create and clear necessary directories
directories = ['out_imgs', 'out_vids', 'out_audio', 'combined_assets', 'final_vids']
for directory in directories:
    create_directory(directory)

all_scenes = []

print("loading agents")
synthetic_agents, helper_agents = instantiate_agents(args.scenario_file_path)
alex = get_agent_by_name("alex", synthetic_agents)
bob = get_agent_by_name("bob", synthetic_agents)
director = get_agent_by_name("director", synthetic_agents)

img_prompt_agent = get_agent_by_name("img_prompt", helper_agents)
vid_prompt_agent = get_agent_by_name("vid_prompt", helper_agents)
audio_prompt_agent = get_agent_by_name("audio_prompt", helper_agents)

character_agents = [alex, bob, director]

observation = "bob walks into the living room and says ' I thought I told you not to drink my beer!'"
all_scenes.append(observation)

print("loading content generation capabilities")
image_gen = FluxWrapper("black-forest-labs/FLUX.1-dev", "lora/cmbnd2.safetensors")
video_gen = VideoWrapper(api="runway")
tts = TTSWrapper(api="eleven_labs")
print("loading complete")

# Simulate the scene
print("simulating scene")
for i in range(args.story_beats):
    for agent in character_agents:
        observation = agent.process_observation(observation, all_scenes, args.story_beats, i)
        all_scenes.append(observation)

print("generating content")
# Generate the content
for i in range(args.variations):
    combined_video_paths = []
    for j in range(args.story_beats):
        print("generating image")
        img_name = f"out_imgs/scene_{i}.png"
        img_text = generate_img_text_for_beat(j, all_scenes)
        img = generate_image(img_text)
        img.save(img_name)
        print("generating video")
        vid_text = generate_vid_text_for_beat(img_text)
        vid_path = generate_video(vid_text, img)
        print("generating VO")
        audio_text = generate_audio_text_for_beat(j, all_scenes)
        audio_path = generate_audio(audio_text)
        combined_output_path = f"combined_assets/scene_{j:04d}_variation_{i:04d}.mp4"
        combine_video_audio(vid_path, audio_path, combined_output_path)
        combined_video_paths.append(combined_output_path)

    print("editing everything together")
    final_video_path = f'final_vids/final_video_variation_{i:04d}.mp4'
    concatenate_videos(combined_video_paths, final_video_path)