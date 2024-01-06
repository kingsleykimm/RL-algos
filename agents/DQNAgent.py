

# steps: episodic step and a normal step
from ..networks.dqn import DQN
from ..utils import *
import random
import torch
class DQNAgent():
    def __init__(self):
        self.network = DQN
        self.target_network = self.network.clone()
        self.config = Config()
        self.replay_buffer = ReplayBuffer(self.config.get_param("replay_size"))
        self.env = AtariEnv(self.config.get_param('game_name'))
    def run(self):
        M = self.config.get_param('episode_steps')
        T = self.config.get_param('max_time')
        initial_e = self.config.get_param('inital_e')
        final_e = self.config.get_param('final_e')
        discount_factor = self.config.get_param('discount_factor')
        exploration_steps = self.config.get_param('exploration_steps')
        annealer = LinearAnnealer(initial_e, final_e, exploration_steps)
        for episode_i in range(M):
            # s1 = {x1}, where x1 is the image
            # initialize sequence, basically get first state
            cur_state, info = self.env.get_first_state()
            for time_step in range(T):
                random_val = random.random()
                action = None
                if random_val <= annealer.incr():
                    action = self.env.action_space.sample()
                    # select a random action
                else:
                    action = torch.argmax(self.network())
                    pass
                    # argmax from the network when inputted the new state
                obsv, reward, term, trunc, info = self.env.step(action)
                self.replay_buffer.add_experience((cur_state, action, reward, obsv))
                cur_state = obsv
                # sample minibatch, but only after replay_start_size is finished
                target = None
                if term or trunc:
                    target = reward
                else:
                    target = reward + discount_factor * # max of the current actions
                