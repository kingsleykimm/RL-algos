

# steps: episodic step and a normal step
from ..networks.dqn import DQN
from ..utils import *
class DQNAgent():
    def __init__(self):
        self.network = DQN
        self.target_network = DQN
        self.config = Config()
        self.replay_buffer = ReplayBuffer(self.config.get_param("replay_size"))

    def episodic_step(self):
        pass
    def step(self):
        pass