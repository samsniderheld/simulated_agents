import yaml
from .base_agent import BaseAgent
from .synthetic_agent import SyntheticAgent

def instantiate_agents(yaml_file):

    synthetic_agents = []
    helper_agents = []

    with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)

    for path in config['synthetic_agents']:
        agent = SyntheticAgent(path)
        synthetic_agents.append(agent)
    
    for path in config['helper_agents']:
        agent = BaseAgent(path)
        helper_agents.append(agent)

    return synthetic_agents, helper_agents

def get_agent_by_name(name, agents):
    for agent in agents:
        if agent.name == name:
            return agent
    
    raise ValueError("Can't find agent with that name")