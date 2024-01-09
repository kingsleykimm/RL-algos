import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class DQN(nn.Module):
    def __init__(self, n_actions=18):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.hidden1 = nn.Linear(7 * 7 * 64, 512) # it is 7 because of the math behind Conv2D, and then 64 out channels
        self.output = nn.Linear(512, n_actions)
        self.apply(self.init_weights)
    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1) # reshapes y to be its first dimension * (everything else squished)
        y = F.relu(self.hidden1(y))
        y = self.output(y)
        return y
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

class DuelingDQN(nn.Module):
    def __init__(self, n_actions=18):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.statelayer = nn.Linear(128, 1)
        self.advantagelayer = nn.Linear(128, n_actions)
    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1) # flatten it in order to get through linear layer
        state_value = F.relu(self.fc1(y))
        state_value = self.statelayer(state_value)
        advantage_value = F.relu(self.fc1(y))
        advantage_value = self.advantagelayer(advantage_value)
        Q = state_value + (advantage_value - advantage_value.mean())
        return Q
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


            
# A2C implements the same network structure as DQN, but it takes a prob distribution over the actions using a softmax.
class A2C(nn.Module):
    def __init__(self, n_actions=18, action_value=True):
        super(A2C, self).__init__()
        self.feature_dim = 256
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.action_value = action_value
        self.fc1 = nn.Linear(9 * 9 * 32, self.feature_dim)
        self.actor = nn.Linear(self.feature_dim, n_actions) # need to put a probability distribution on this
        if action_value:
            self.critic = nn.Linear(self.feature_dim, n_actions)
        else:    
            self.critic= nn.Linear(self.feature_dim, 1) # value state function
    def forward(self ,x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc1(y))
        policy = self.actor(y)
        state = self.critic(y)
        if self.action_value:
            return policy, state
        return policy, torch.squeeze(state)
    def policy_action(self, state):
        policy, state_value = self(state)
        distribution = F.softmax(policy, dim=-1)
        cat = Categorical(distribution)
        action = cat.sample()
        if self.action_value:
            return action, cat.log_prob(action), state_value[action]
        return action, cat.log_prob(action), state_value

        
