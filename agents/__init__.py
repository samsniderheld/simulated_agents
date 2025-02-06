import yaml
from .base_agent import BaseAgent
from .synthetic_agent import SyntheticAgent
from .shot_agent import ShotAgent

def instantiate_agents(yaml_file):

    synthetic_agents = []
    helper_agents = []

    with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)

    for agent in config['agents']:
        agent = SyntheticAgent(agent['file'])
        agent.load_observations(agent['base_observations'])
        synthetic_agents.append(agent)
    
    for agent in config['helpers']:
        agent = BaseAgent(agent['file'])
        helper_agents.append(agent)

    return synthetic_agents, helper_agents

def get_agent_by_name(name, agents):
    for agent in agents:
        if agent.name == name:
            return agent
    return None