import torch.nn as nn
import torch
import torch.nn.functional as F

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
