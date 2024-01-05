import numpy as np
import random
from collections import deque

class ReplayBuffer(object):
    def __init__(self, N):
        self.replay = deque()
        self.capacity = N
    def add_experience(self, experience):
        if len(self.replay) == self.capacity:
            self.replay.popleft()
        self.replay.append(experience)
    def sample(self, batch_size):

        return random.sample(list(self.replay), batch_size)