import torch
from utils import *
from networks import *
class PPOAgent():
    def __init__(self, config):
        self.network = PPOActorCritic()
        self.config = config
        self.env = AtariVecEnv(self.config.env_name, self.config.num_envs)
        self.optimizer = self.config.optimizer(self.network.parameters())
        self.storage = MultiEnvRollouts(self.config.rollout_length)
    def run(self):
        next_obs, _ = self.env.get_first_state()
        next_dones = [0] * self.config.num_envs
        for update in range(1, self.config.max_time // (self.num_envs * self.rollout_length)):
            rollout_entropy = 0
            for i in range(self.rollout_length):
                obs = next_obs
                done = next_dones
                # get actions
                actions, log_probs, critics, entropy = self.network.get_batch_actions(obs) # entropy is normalized here per step
                next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)
                next_dones = terminations or truncations
                self.storage.add_step((actions, log_probs, critics, rewards, next_dones))
                rollout_entropy += entropy
    def learn(self, final_value, dones, entropy):
        

