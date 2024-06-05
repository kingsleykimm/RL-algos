from utils import *
from networks import *


class DDPGAgent():
    def __init__(self, config):
        self.config = config
        self.q_network = DQN()
        self.target_q_network = DQN()
        self.target_q_network.load_state_dict(self.network.state_dict())
        self.replay_buffer = ReplayBuffer(config.replay_size)
        self.env = AtariEnv(config.game_name)
        self.q_optimizer = self.config.optimizer(self.q_network.parameters())
        self.q_updates = 0
        self.policy_updates = 0
        