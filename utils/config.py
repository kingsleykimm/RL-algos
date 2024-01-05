
import argparse
import torch
class Config:
    DEVICE = torch.device('cpu')
    DEFAULT_REPLAY = 'experience_replay'
    def __init__(self):
        self.config_map = {}
        self.config_map['minibatch_size'] = 32
        self.config_map['replay_size'] = 1e8
        self.config_map['n_step'] = 4
        self.config_map['network_update_freq'] = 1e5
        self.config_map['discount_factor'] = 0.99
        self.config_map['initial_e'] = 1
        self.config_map['final_e'] = 0.1
        self.config_map['final_exploration'] = 1e8
        self.config_map['noop_max'] = 10
        self.config_map['replay_start_size'] = 50000
    def change_param(self, param_name, new_amount):
        if param_name not in self.config_map:
            return KeyError
        self.config_map[param_name] = new_amount
    def get_param(self, param_name):
        if param_name not in self.config_map:
            return KeyError
        return self.config_map[param_name]