import numpy as np
import random
from collections import deque
import torch
from torch.utils.data import Dataset
class OldReplayBuffer(object):
    def __init__(self, N):
        self.replay = deque()
        self.capacity = N
    def add_experience(self, experience):
        if len(self.replay) == self.capacity:
            self.replay.popleft()
        self.replay.append(experience)
    def sample(self, batch_size=1):

        return random.sample(list(self.replay), batch_size)

class ReplayBuffer(Dataset):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))
        
    def __getitem__(self, index):
        return self.buffer[index]
    
    def sample(self, batch_size=32):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for state, action, reward, next_state, done in samples:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

    # could add the compute loss function here, but it's better to modularize in the A2C/A3C classes

# class MultiEnvRollouts():
#     def __init__(self, size, num_envs):
#         self.capacity = size
#         self.num_envs = num_envs
#         self.values = [[] for _ in range(num_envs)]
#         self.log_probs = [[] for _ in range(num_envs)]
#         self.dones = [[] for _ in range(num_envs)]
#         self.rewards = [[] for _ in range(num_envs)]
#         self.actions = [[] for _ in range(num_envs)]
#     def add_step(self, step):
#         actions, log_probs, values, rewards, dones = step

        
        

