import argparse

from utils import *
from agents import *
from content_generation import *

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Simulated Agents")
parser.add_argument('--iterations', type=int, default=3, help='Number of story beats')
parser.add_argument('--scenario_file_path', type=str, default='config_files/scenario.yaml', help='Path to the scenario file')
parser.add_argument('--variations', type=int, default=1, help='Number of variations')
parser.add_argument('--interactive', action='store_true', help='Interactive mode')
parser.add_argument('--narrative', type=str, default='today we are writing a story about a young boy learning about the universe', help='what are we writing an episode about')

args = parser.parse_args()

all_scenes = []

# Create and clear necessary directories
directories = ['out_imgs', 'out_vids', 'out_audio', 'combined_assets', 'final_vids']
for directory in directories:
    create_directory(directory)

#load all the agents
print("loading agents")
synthetic_agents, helper_agents = instantiate_agents(args.scenario_file_path)
script_writer = get_agent_by_name("script_writer", synthetic_agents)
producer = get_agent_by_name("producer", synthetic_agents)
img_prompt_agent = get_agent_by_name("img_prompt", helper_agents)
vid_prompt_agent = get_agent_by_name("vid_prompt", helper_agents)

character_agents = [script_writer, producer]

#load all the content generation capabilities
print("loading content generation capabilities")
image_gen = FluxWrapper("black-forest-labs/FLUX.1-dev", ["lora/ARCANE_STYLE_FADOO-FLUX.safetensors", "lora/taylrrdect-v4.safetensors"])
video_gen = VideoWrapper(api="kling")
tts = TTSWrapper(api="eleven_labs")
print("loading complete")

#instantiate the first observation
observation = args.narrative
all_scenes.append(observation)
final_script = None

# Simulate the scene
print("simulating scene")
for i in range(args.iterations):
    for agent in character_agents:
        if agent.name == "script_writer":
            if i == 0:
                script = script_writer.process_observation(observation, all_scenes, use_structured=True)
            else:
                script = script_writer.process_observation(f"please make the following changes to the orignal script: {observation}", all_scenes, use_structured=True)
            
            script_str = script.to_str()
            print(script_str)
            all_scenes.append(script_str)
        
        elif agent.name == "producer":

            observation = producer.process_observation(f"what do you think of : {script_str} tell the script writer what they should change", all_scenes)
            print(observation)
            all_scenes.append(observation)

final_script = script

script_len = len(final_script.shots)

# Generate the content
print("generating content")
for i in range(args.variations):
    combined_video_paths = []
    for j,shot in enumerate(final_script.shots):
        print("generating image")
        img_name = f"out_imgs/scene_{i}.png"

        augmented_prompt = img_prompt_agent.basic_api_call(shot.txt2img_prompt)
        img_text = f"{script_writer.lora_key_word},\n\n {augmented_prompt}, \n\n Costume: {script_writer.flux_caption}"
        img = image_gen.generate_image(img_text)
        img.save(img_name)
        print("generating video")
        vid_text = vid_prompt_agent.basic_api_call(shot.txt2img_prompt)
        vid_path = video_gen.make_api_call(vid_text, img, duration=10)
        print("generating VO")
        audio_text = shot.vo
        audio_path = tts.make_api_call(audio_text)
        combined_output_path = f"combined_assets/scene_{j:04d}_variation_{i:04d}.mp4"
        combine_video_audio(vid_path, audio_path, combined_output_path)
        combined_video_paths.append(combined_output_path)

    print("editing everything together")
    final_video_path = f'final_vids/final_video_variation_{i:04d}.mp4'
    concatenate_videos(combined_video_paths, final_video_path)