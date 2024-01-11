import torch
from utils import *
from networks import *

class A2CAgent():
    def __init__(self, config):
        self.config = config
        self.network = ActorCritic(action_value=False)
        self.env = AtariEnv(config.game_name)
        self.storage = EpisodeRollout(self.config.rollout_length)
        self.optimizer = self.config.optimizer(self.network.parameters())
        # need to put an annealer here
        self.entropy_coeff = self.config.entropy_coeff
    def run(self):
        num_episodes = self.episode_steps
        for episode_ind in range(num_episodes):
            val, entropy = self.rollout_episode(self.config.rollout_length)
            loss = self.compute_loss(val, entropy)
            loss.backward()
            self.optimizer.zero_grad()
            self.optimizer.step()
    def rollout_episode(self, episode_length):
        state = self.env.get_first_state()
        action = self.network.policy_action(state)
        episode_entropy = 0
        # experience pair: (SARSA): (state, action, reward, s')
        for t in range(len(episode_length)):
            next_state, reward, term, trunc, info = self.env.step(action)
            next_action, log_prob, value, entropy = self.network.policy_action(next_state) # value here can either be Q(s, a) or V(s)
            episode_entropy += entropy
            self.storage.add_step((reward, log_prob, value, term or trunc))
            state = next_state
        with torch.no_grad():
            _, _, final_value = self.network.policy_action(state) # final_value used for last step when doing episode rollout
        return final_value, episode_entropy
    def compute_loss(self, final_value, entropy):
        # final_value : final state/action value taken
        q_vals = [0] * len(self.storage.values) # Q(s, a)
        q = final_value
        for i in range(len(self.storage.values) - 1, -1, -1):
            q = self.storage.rewards[i] + self.config.discount_factor * (1 - self.storage.dones[i]) * q
            q_vals[i] = q
        advantage = torch.Tensor(q_vals) - torch.Tensor(self.storage.values) # Q(s, a) - V(s) for all time steps | or using the equation above since we only have Value functions
        critic_loss = advantage.pow(2).mean()
        log_probs = -1 * torch.Tensor(self.storage.log_probs)
        actor_loss = torch.mul(log_probs, advantage.detach()).mean()
        return critic_loss + actor_loss + entropy * self.entropy_coeff

        