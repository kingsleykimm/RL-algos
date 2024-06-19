from agents import *
import torch
from utils import *

config = Config()
env = MujocoEnv("Ant-v4")
device = None
if torch.cuda.is_available():
    device = torch.cuda.current_device()
agent = SAC(config, env, env.obs_space, env.action_space, device)
agent.train()

