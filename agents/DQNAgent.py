

# steps: episodic step and a normal step
from ..networks.dqn import DQN
from ..utils import *
class DQNAgent():
    def __init__(self):
        self.network = DQN
        self.target_network = DQN
        self.config = Config()
        self.replay_buffer = ReplayBuffer(self.config.get_param("replay_size"))
        self.env = make_atari_env(self.config.get_param('game_name'))
    def run(self):
        M = self.config.get_param('episode_steps')
        T = self.config.get_param('max_time')
        initial_e = self.config.get_param('inital_e')
        final_e = self.config.get_param('final_e')
        exploration_steps = self.config.get_param('exploration_steps')
        annealer = LinearAnnealer(initial_e, final_e, exploration_steps)
        
        pass