import torch
from utils import *
from networks import *

class A2CAgent():
    def __init__(self, config):
        self.config = config
        self.network = A2C()
        self.env = AtariEnv(config.game_name)

    def run(self):
        max_time = self.config.max_time
        num_episodes = self.episode_steps
        state = self.env.get_first_state()
        for t in range(max_time):
            