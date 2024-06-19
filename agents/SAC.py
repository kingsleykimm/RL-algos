from utils import *
from networks import *
from torch.utils.data import DataLoader
from torch.optim import Adam
import logging

logger = logging.Logger("SAC LOGGER")
class SAC():
    def __init__(self, config, 
                 env : MujocoEnv,
                 num_states, # environment / state / observation size
                  num_actions,
                  device):
        # off policy SAC
        self.config = config
        self.device = device
        self.replay_buffer = ReplayBuffer(self.config.replay_size)
        self.sampler = DataLoader(self.replay_buffer)
        self.value_network = ValueNetwork(num_states).to(device) # remember to put in action scaling
        self.target_value_network = ValueNetwork(num_states).to(device)
        self.target_value_network.load_state_dict(self.value_network.state_dict())
        self.q_networks = QNetwork(num_states, num_actions).to(device)
        self.policy = GuassianPolicy(num_actions, num_states).to(device)
        self.env : MujocoEnv = env
        self.target_update = 1
        self.step_counter = 0
        self.action_space = env.action_space
        # Optimizers
        self.value_optim = Adam(self.value_network.parameters(), lr=self.config.lr)
        self.q_optim = Adam(self.q_networks.parameters(), lr=self.config.lr)
        self.actor_optim = Adam(self.policy.parameters(), lr=self.config.lr)
    def train(self):
        
        first_state, info = self.env.get_first_state()
        cur_state = first_state
        for i in range(int(self.config.exploration_steps)): # 3 initial episodes to fill buffer
            rand_action = self.random_action()
            obs, reward, term, trunc, info = self.env.step(rand_action)
            self.replay_buffer.add(torch.from_numpy(cur_state), rand_action, reward, torch.from_numpy(obs), term)
            cur_state = obs
            if term or trunc:
                cur_state, info = self.env.get_first_state()
                continue


        while self.step_counter < self.config.env_steps:
            
            # new_state, means get starting state
            first_state, info = self.env.get_first_state()
            cur_state = first_state
            # collect for n environment steps
            while True:
                self.step_counter += 1
                
                print("Iteration: ", self.step_counter)
                action, log_prob, linear_output = self.select_action(cur_state)
                print(action)
                obs, reward, term, trunc, info = self.env.step(action)
                self.replay_buffer.add(torch.from_numpy(cur_state), action, reward, torch.from_numpy(obs), term)
                if self.step_counter == self.config.env_steps:
                    break
                if term or trunc:
                    cur_state = self.env.get_first_state()
                    break
                else:
                    cur_state = obs
            # gradient steps
            for step in range(self.config.gradient_steps):
                self.calculate_loss()
                self.update_counter += 1


    def select_action(self, state):
        return self.policy.sample(state)
    def random_action(self):
        rand_action = torch.empty(self.action_space)
        for ind, value in enumerate(rand_action):
            rand_action[ind] = random.uniform(-1, 1)
        return rand_action
    def calculate_loss(self):
        print("Calculating Loss")
        states, actions, rewards, next_states, mask_batch = self.replay_buffer.sample(batch_size=self.config.batch_size)
        print(mask_batch)
        mask_batches = torch.tensor(mask_batch) # this is the dones, we don't want to train on ending states
        with torch.no_grad():
            next_state_values = self.value_network(states)
            q_value_target = rewards + mask_batch * (self.config.gamma * next_state_values)
        x1, x2 = self.q_networks(states, actions)
        q_loss = nn.MSELoss(x1, q_value_target) + nn.MSELoss(x2, q_value_target)
        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()
        min_q = min(x1, x2)
        # value loss
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.policy.sample(next_states)
            value_target = min_q - self.config.alpha * next_log_probs
        value_loss = nn.MSELoss(self.value_network(states), value_target)
        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()
        # policy loss
        policy_loss = (self.config.alpha * next_log_probs - min_q).mean() # mean because of reparameterization trick, removing variance/noise
        self.actor_optim.zero_grad()
        self.actor.backward()
        self.actor_optim.step() 

        if self.update_counter % self.target_update == 0:
            self.target_value_network.parameters = (1 - self.config.tau) * self.target_value_network.parameters + self.config.tau * self.value_network.parameters
        pass



    def update_parameters(self):
        pass