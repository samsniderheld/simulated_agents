# Simulated Agents

This repository contains code for creating and managing simulated agents using language models. The agents can handle synthetic memory, context updates, and script writing.

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

## LLM Configuration

Before running the agents, you need to configure the language model settings. Update the `llm_wrapper.py` file with your OpenAI API key and other settings.


## Agent Cofiguration

Agents can easily be configured using .json files.

Example `context_agent.json`:

```json
{
    "llm": "openAI",
    "system_prompt": "Your name is bob. You are shy, funny, and a bit neurotic.",
    "memory_limit": 5
}
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

### ContextAgent
The `ContextAgent` class updates and manages context.

#### Methods:
- `update_context(context: List[str]) -> str`: Updates the context based on the provided context list.

### ScriptAgent
The `ScriptAgent` class writes scripts based on context.

#### Methods:
- `write_script(context: List[str]) -> str`: Writes a script based on the provided context list.

## Example

Here is an example of how to use the `SyntheticAgent`:

```python
from agents import SyntheticAgent

# Initialize the agent with a configuration file
agent = SyntheticAgent(config_file='config_files/context_agent.json')

# Add observations to the agent's memory
agent.add_to_memory("The sky is clear.")
agent.add_to_memory("Birds are singing.")

# Summarize the memory
agent.summarize_memory()

# Reflect on the memory
reflection = agent.reflect()
print(reflection)
```

