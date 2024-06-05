import gymnasium as gym
import numpy as np
import torch
import dm_env


class MujocoEnv():
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.obs_space = self.env.observation_space
    def get_first_state(self):
        cur, info = self.env.reset()
        return cur, info
    def step(self, action):
        action = torch.Tensor.numpy(action)
        return self.env.step(action)
class AtariEnv():
    def __init__(self, env_name):
        self.env = gym.make(env_name, obs_type="rgb", frameskip=1)
        self.env = gym.wrappers.AtariPreprocessing(self.env, grayscale_obs=False, frameskip=4)
        self.env = BaseWrapper(self.env)
    def get_first_state(self):
        cur, info = self.env.reset()
        return torch.from_numpy(cur), info
    def random_action(self):
        return self.env.action_space.sample()
    def step(self, action):
        action = torch.Tensor.numpy(action)
        return self.env.step(action)
    
class AtariVecEnv():
    def __init__(self, env_name, num_envs):
        self.envs = gym.vector.make(env_name, frameskip=1, num_envs=num_envs, wrappers=[gym.wrappers.AtariPreprocessing, BaseWrapper])
    def get_first_state(self):
        states, info = self.envs.reset()
        return torch.from_numpy(states), info
    def random_action(self):
        return self.envs.single_action_space.sample()
    def step(self, actions):
        actions = torch.Tensor.numpy(actions)
        observations, rewards, terminations, trunactions, infos = self.envs.step(actions)
        return observations, rewards, terminations, trunactions, infos

class BaseWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.total_rewards = 0
    def step(self, action):
        obsv, reward, term, trunc, info = self.env.step(action)
        if term or trunc:
            info['episodic_return'] = self.total_rewards
            self.env.reset()
            self.total_rewards = 0
        else:
            self.total_rewards += reward
            info['episodic_return'] = None
        return torch.from_numpy(obsv), reward, term, trunc, info
