import torch.nn as nn
import argparse
import torch
class Config:
    DEVICE = torch.device('cpu')
    DEFAULT_REPLAY = 'experience_replay'
    def __init__(self):
        self.batch_size = 32
        self.replay_size = 10e6
        self.n_step = 4
        self.network_update_freq = 1e5
        self.discount_factor = 0.99
        self.initial_e = 1
        self.final_e = 0.1
        self.exploration_steps = 1e4
        self.noop_max = 30
        self.replay_start_size = 50000
        self.game_name = 'ALE/Adventure-v5'
        self.episode_steps = 2000
        self.max_time = 1000
        self.rollout_length = 5
        self.lr = 3e-4
        self.tau = 0.005
        self.update_interval = 1
        self.gradient_steps = 1
        self.env_steps = 3e6
