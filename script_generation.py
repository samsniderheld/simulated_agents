import argparse
import json

from utils import *
from agents import *
from content_generation import *

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Simulated Agents")
parser.add_argument('--iterations', type=int, default=3, help='Number of story beats')
parser.add_argument('--scenario_file_path', type=str, default='config_files/scenario.yaml', help='Path to the scenario file')
parser.add_argument('--interactive', action='store_true', help='Interactive mode')
parser.add_argument('--augment_prompts', action='store_true', help='Interactive mode')

parser.add_argument('--narrative', type=str, default='today we are writing a story about a young boy learning about the universe', help='what are we writing an episode about')

args = parser.parse_args()

all_scenes = []


#load all the agents
print("loading agents")
synthetic_agents, helper_agents = instantiate_agents(args.scenario_file_path)
script_writer = get_agent_by_name("script_writer", synthetic_agents)
producer = get_agent_by_name("producer", synthetic_agents)
img_prompt_agent = get_agent_by_name("img_prompt", helper_agents)

character_agents = [script_writer, producer]


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
                script = script_writer.process_observation(observation, all_scenes, args.iterations, i, use_structured=True)
            else:
                script = script_writer.process_observation(f"please make the following changes to the orignal script: {observation}", all_scenes, args.iterations, i, use_structured=True)
            
            script_str = script.to_str()
            print(script_str)
            all_scenes.append(script_str)
        
        elif agent.name == "producer":

            observation = producer.process_observation(f"what do you think of : {script_str} tell the script writer what they should change", all_scenes, args.iterations, i)
            print(observation)
            all_scenes.append(observation)

final_script = script_writer.process_observation(f"please make the following changes to the orignal script: {observation}", all_scenes, args.iterations, i, use_structured=True)

if args.augment_prompts:
    print("augmenting prompts")
    for i,shot in enumerate(final_script.shots):

        img_prompt = img_prompt_agent.basic_api_call(shot.txt2img_prompt)
        final_script.shots[i].txt2img_prompt = img_prompt

script_json = final_script.to_json() 

with open('script.json', 'w') as json_file:
    json.dump(script_json, json_file, indent=4)

print("script saved to script.json")


