from utils import *
from networks import *
from torch.utils.data import DataLoader

class SAC():
    def __init__(self, config, 
                 env : MujocoEnv,
                 num_states, # environment / state / observation size
                  num_actions):
        # off policy SAC
        self.config = config
        self.replay_buffer = ReplayBuffer(self.config.replay_size)
        self.sampler = DataLoader(self.replay_buffer)
        self.value_network = ValueNetwork(num_states) # remember to put in action scaling
        self.q_networks = QNetwork(num_states, num_actions)
        self.policy = GuassianPolicy(num_actions)
        self.env : MujocoEnv = env
    def train(self):
        for counter in range(self.config.iterations):
            # new_state, means get starting state
            first_state, info = self.env.get_first_state()
            cur_state = first_state
            # collect for n environment steps
            for step in range(self.config.environment_steps):
                action, log_prob, linear_output = self.select_action(cur_state)
                obs, reward, term, trunc, info = self.env.step(action)
                self.replay_buffer.add(cur_state, action, reward, obs, log_prob)
                if term or trunc:
                    cur_state = self.env.get_first_state()
                else:
                    cur_state = obs
            # gradient steps
            for step in range(self.config.gradient_steps):
        


    def select_action(self, state):
        return self.policy.sample(state)

    def calculate_loss(self):
        states, actions, rewards, next_states = self.replay_buffer.sample(batch_size=self.config.batch_size)
        states = torch.tensor(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.tensor(next_states)
        x1, x2 = self.q_networks(states, actions)
        pass

    def hard_update(self):
        pass
    def soft_update(self):
        pass

    def update_parameters(self):
        pass