import gymnasium as gym
import numpy as np
import torch

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.atari_wrappers import FrameStack as FrameStack
env = gym.make(obs_type="rgb")