from agents import DQNAgent
import torch
from utils import *

config = Config()
config.optimizer = lambda params : torch.optim.RMSprop(params, lr=2.5e-4, momentum=0.95, eps=0.01, alpha=0.95)
agent = DQNAgent(config)
agent.run()

