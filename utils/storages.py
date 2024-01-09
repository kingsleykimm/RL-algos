import numpy as np
import random
from collections import deque
import torch

class ReplayBuffer(object):
    def __init__(self, N):
        self.replay = deque()
        self.capacity = N
    def add_experience(self, experience):
        if len(self.replay) == self.capacity:
            self.replay.popleft()
        self.replay.append(experience)
    def sample(self, batch_size=1):

        return random.sample(list(self.replay), batch_size)

class EpisodeRollout():
    def __init__(self, size):
        self.capacity = size
        self.values = []
        self.log_probs = []
        self.dones = []
        self.rewards = []
    def add_step(self, step):
        reward, log_prob, value, done = step
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
    # could add the compute loss function here, but it's better to modularize in the A2C/A3C classes