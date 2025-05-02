import numpy as np

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def get_actions(self, obs):
        # Simply sample random actions for each agent.
        return self.action_space.sample()