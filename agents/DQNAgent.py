

# steps: episodic step and a normal step
from ..networks.dqn import DQN
from ..utils import *
import random
class DQNAgent():
    def __init__(self):
        self.network = DQN
        self.target_network = DQN
        self.config = Config()
        self.replay_buffer = ReplayBuffer(self.config.get_param("replay_size"))
        self.env = AtariEnv(self.config.get_param('game_name'))
    def run(self):
        M = self.config.get_param('episode_steps')
        T = self.config.get_param('max_time')
        initial_e = self.config.get_param('inital_e')
        final_e = self.config.get_param('final_e')
        exploration_steps = self.config.get_param('exploration_steps')
        annealer = LinearAnnealer(initial_e, final_e, exploration_steps)
        for episode_i in range(M):
            # initialize sequence, basically get first state
            for time_step in range(T):
                random_val = random.random()
                if random_val <= annealer.incr():
                    # select a random action
                else:
                    # argmax from the network when inputted the new state

        pass