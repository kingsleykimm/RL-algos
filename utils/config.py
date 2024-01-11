import torch.nn as nn
import argparse
import torch
class Config:
    DEVICE = torch.device('cpu')
    DEFAULT_REPLAY = 'experience_replay'
    def __init__(self):
        self.minibatch_size = 32
        self.replay_size = 1e8
        self.n_step = 4
        self.network_update_freq = 1e5
        self.discount_factor = 0.99
        self.initial_e = 1
        self.final_e = 0.1
        self.exploration_steps = 1e8
        self.noop_max = 30
        self.replay_start_size = 50000
        self.game_name = 'ALE/Adventure-v5'
        self.episode_steps = 2000
        self.max_time = 1000
        self.optimizer = None
        self.actor_threads = 16
        self.rollout_length = 5
        self.actor_lr = None
        self.policy_lr = None
        self.entropy_coeff = None
        self.num_envs = None
        self.surrogate_clip = None