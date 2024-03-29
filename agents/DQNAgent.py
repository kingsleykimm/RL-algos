

# steps: episodic step and a normal step
from networks import *
from utils import *
import random
import torch
class DQNAgent():
    def __init__(self, config):
        self.config = config
        self.network = DQN()
        self.target_network = DQN()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay_buffer = ReplayBuffer(self.config.replay_size)
        self.env = AtariEnv(self.config.game_name)
        self.optimizer = self.config.optimizer(self.network.parameters())
        self.param_updates = 0
    def run(self):
        M = self.config.episode_steps
        T = self.config.max_time
        network_update = self.config.network_update_freq
        initial_e = self.config.initial_e
        final_e = self.config.final_e
        self.discount_factor = self.config.discount_factor
        exploration_steps = self.config.exploration_steps
        batch_size = self.config.minibatch_size
        annealer = LinearAnnealer(initial_e, final_e, exploration_steps)
        self.populate_memory()
        print('Memory populated, starting training')
        for episode_i in range(M):
            # s1 = {x1}, where x1 is the image
            # initialize sequence, basically get first state
            cur_state, info = self.env.get_first_state()
            for time_step in range(T):
                random_val = random.random()
                action = None
                if random_val <= annealer.incr():
                    action = self.env.random_action()
                    # select a random action
                else:
                    action = torch.argmax(self.network(cur_state))
                    # argmax from the network when inputted the new state
                print(action)
                obsv, reward, term, trunc, info = self.env.step(action)
                self.replay_buffer.add_experience((cur_state, action, reward, obsv, term or trunc))
                cur_state = obsv
                # sample minibatch, but only after replay_start_size is finished
                if time_step % 4 == 0: #update frequency
                    minibatch = self.replay_buffer.sample(batch_size)
                    loss = self.get_batch_loss(minibatch)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.param_updates += 1
                    if self.param_updates % network_update == 0: #every C steps, updates the network
                        self.target_network.load_state_dict(self.network.state_dict())
    def get_batch_loss(self, batch):
        target = torch.Tensor()
        q = torch.Tensor()
        for experience in batch:
            s_j, action_j, reward_j, s_j_1, end = experience    
            if end:
                target.add(reward_j)
            else:
                with torch.no_grad():
                    target.add(reward_j + self.discount_factor * max(self.target_network(s_j_1)))
            q.add(self.network(s_j)[action_j])
        loss = (target - q).square().mean()
        return loss


    def populate_memory(self):
        start_size = self.config.replay_start_size
        counter = 0
        cur_state, info = self.env.get_first_state()
        while counter < start_size // 4:
            action = self.env.random_action()
            obsv, reward, term, trunc, info = self.env.step(action)
            self.replay_buffer.add_experience((cur_state, action, reward, obsv))
            cur_state = obsv
            if term or trunc:
                cur_state, info = self.env.reset()