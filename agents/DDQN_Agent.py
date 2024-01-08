# Double Q-Learning Agent
from .DQNAgent import DQNAgent
from utils import *
import random
import torch
class DDQNAgent(DQNAgent):
    def __init__(self, config):
        super().__init__(self, config)
    def get_batch_loss(self, batch):
        for experience in batch:
            target = torch.Tensor()
            q = torch.Tensor()
            for experience in batch:
                s_j, action_j, reward_j, s_j_1, end = experience    
                if end:
                    target.add(reward_j)
                else:
                    with torch.no_grad():
                        val = reward_j + self.discount_factor * self.network(s_j_1)[torch.argmax(self.target_network(s_j_1))]
                q.add(self.network(s_j)[action_j])
        loss = (target - q).square().mean()
        return loss
        