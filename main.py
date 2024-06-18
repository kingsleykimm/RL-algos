from agents import *
import torch
from utils import *

config = Config()
env = MujocoEnv("Ant-v4")
agent = SAC(config, env, env.obs_space, env.action_space)
agent.train()

