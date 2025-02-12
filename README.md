# Simulated Agents

This repository contains code for creating and managing simulated agents using language models. The agents can handle synthetic memory, context updates, script writing, and more.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/samsniderheld/simulated_agents.git
    cd simulated_agents
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Install extra dependencies:

    ```sh
    pip install torchao --extra-index-url https://download.pytorch.org/whl/cu121 # full options are cpu/cu118/cu121/cu124
    pip install git+https://github.com/xhinker/sd_embed.git@main
    ```

## API Configurations

Before running the agents, you need to configure the your APIs. Here is the OpenAI Example.

1. Obtain your OpenAI API key from the [OpenAI website](https://beta.openai.com/signup/).

2. Set your OpenAI API key as an environment variable:
    ```sh
    export OPENAI_API_KEY='your_openai_api_key'
    ```

## Agent Configuration

Agents can easily be configured using YAML files.

Example `alex.yaml`:

```yaml
name: alex,
system_prompt: Your name is alex. You are passive aggressive, controlling, and inconsiderate,
lora_key_word: ALX2,
flux_caption: ALX2 is wearing a dark green sweater and black pants. He has medium length blond hair.
```

## Usage

### BaseAgent
The `BaseAgent` class is the base class for all agents. It loads the configuration file and initializes the language model wrapper.

### SyntheticAgent
The `SyntheticAgent` class handles synthetic memory and processing.

#### Methods:
- `add_to_memory(observation: str)`: Adds an observation to short-term memory.
- `summarize_memory()`: Summarizes the short-term memory and appends the summary to long-term memory.
- `reflect() -> str`: Reflects on the long-term and short-term memory to determine feelings.
- `process_observation(observation: str, context: str, num_beats: int, current_beat: int) -> str`: Processes an observation within the given context and story beats.
- `load_observations(observations: List[str])`: Loads multiple observations into short-term memory.

## Example

Here is an example of how to use the `SyntheticAgent`:

```python
from utils import *
from agents import *
story_beats = 6

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

shot_agent = ShotAgent()

character_agents = [alex,bob]

scene = []

observation = "bob walks into the living room and says ' I thought I told you not to drink my beer!'"
scene.append(observation)

print(f"{observation}\n\n")
for i in range(story_beats):
for agent in character_agents:
    observation = agent.process_observation(observation, scene,story_beats,i)
    print(f"{observation}\n\n")
    scene.append(observation)

bob.summarize_memory()
alex.summarize_memory()
```

Here is an example of how to run a basic example via gradio:

## Usage
The script supports the following command-line arguments:

- `--iterations`: Number of iterations for the simulation (default: 3)
- `--scenario_file_path`: Path to the scenario file (default: 'config_files/scenario.yaml')
- `--share`: Share the Gradio app (default: False)
- `--debug`: Debug mode (default: False)
- `--interactive`: Interactive mode (default: False)
- `--show_simulated_thinking`: Show the simulated thinking (default: False)
- `--narrative`: Narrative for the episode (default: 'today we are writing a story about a young boy learning about the universe')

Example usage:

    ```sh
    python gradio_interface.py --iterations 5 --scenario_file_path config_files/scenario.yaml --share --narrative "A story about a brave knight on a quest"
    ```

This will run the `gradio_interface.py` script with 5 iterations, using the specified scenario file, sharing the Gradio app, and using the provided narrative.


Here is an example of how to run a full agentic flow:
## Usage
The script supports the following command-line arguments:

- `--iterations`: Number of iterations for the simulation (default: 3)
- `--scenario_file_path`: Path to the scenario file (default: 'config_files/scenario.yaml')
- `--variations`: Number of variations for the generated content (default: 1)
- `--interactive`: Interactive mode (default: False)
- `--narrative`: Narrative for the episode (default: 'today we are writing a story about a young boy learning about the universe')

Example usage:

    ```sh
    python full_agentic_flow.py --iterations 5 --scenario_file_path config_files/scenario.yaml --variations 2 --narrative "A story about a brave knight on a quest"
    ```

This will run the `full_agentic_flow.py` script with 5 iterations, using the specified scenario file, generating 2 variations of the content, and using the provided narrative.
    ```
