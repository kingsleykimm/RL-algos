import torch
from utils import *
from networks import *
class PPOAgent():
    def __init__(self, config):
        self.network = PPOActorCritic()
        self.config = config
        self.env = AtariVecEnv(self.config.env_name, self.config.num_envs)
        self.optimizer = self.config.optimizer(self.network.parameters())
        self.storage = EpisodeRollout(self.config.rollout_length)
    def run(self):
        next_obs, _ = self.env.get_first_state()
        next_dones = [0] * self.config.num_envs
        for update in range(1, self.config.max_time // (self.num_envs * self.rollout_length)):
            data = []
            rollout_entropy = 0
            for i in range(self.rollout_length):
                obs = next_obs
                done = next_dones
                # get actions
                actions, log_probs, critics, entropy = self.network.get_batch_actions(obs) # entropy is normalized here per step
                next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)
                next_dones = terminations or truncations
                self.storage.add_step((rewards, log_probs, critics, next_dones))
                rollout_entropy += entropy
            with torch.no_grad():
                _, critic = self.network(next_obs)
            # flatten the batch from M states of N environments into N * M states
            self.learn(critic, rollout_entropy)
    def learn(self, final_values, entropy):
        # need to shuffle and create mini-batches first, with update_epochs
        rewards = self.storage.rewards.reshape(-1)
        log_probs = self.storage.log_probs.reshape(-1)
        dones = self.storage.dones.reshape(-1)
        values = self.storage.values.reshape(-1)
        batch_inds = np.arange(len(rewards)) # reshape to N * M
        batch_size = len(rewards)
        minibatch_size = batch_size // self.config.minibatch_n
        for i in range(self.update_epochs):
            np.random.shuffle(batch_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                inds = batch_inds[start:end]
                
        # Generalized Advantage Estimation
        q_vals = torch.Tensor()
        q = final_values # this is already a tensor
        for i in range(len(self.num_envs) - 1, -1, -1):
            done_effect = torch.mul(-1, self.storage.dones[i]) + torch.ones(self.num_envs)
            q = self.storage.rewards[i] + self.discount_factor * torch.mul(q, done_effect)
            q_vals.add(q)
        torch.flip(q_vals, dim=-1)
        advantages = q_vals - self.storage.values
            




