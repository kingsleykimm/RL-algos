import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical

from abc import ABC, abstractmethod

def weights_init(m):
    def init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    return init

class AtariNetworkHead(nn.Module):
    def __init__(self):
        super(AtariNetworkHead, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

class DeterministicPolicy(nn.Module):
    def __init__(self, n_actions, input_dim, hidden_dim=256, action_space=None):
        super(DeterministicPolicy, self).__init__()
        # 2 hidden layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

        # noise tensor to add to actions at the end, when sampling
        self.noise = torch.Tensor(n_actions)
        self.apply(weights_init)
        if action_space == None:
            self.action_space = 1.
            self.action_bias = 0.
        else:
            self.action_space = (action_space.high - action_space.low) / 2. # range bounds
            self.action_bias = (action_space.high + action_space.low) / 2. # bias is average
    def forward(self, x):
        x = torch.tensor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x * self.action_scale + self.action_bias 
        # scale the action + bias it, since tanh is between -1 and 1, since it's centered around 0, and then move it towards the bias
    # use reparameterization trick here
    def sample(self, input_state):
        input_state = torch.tensor(input_state)
        x = self.forward(input_state)
        noise = self.noise.normal_(0., 0.1)
        noise = noise.clamp(-0.25, 0.25)
        return x + noise, x
class GuassianPolicy(nn.Module):
    def __init__(self, n_actions, input_dim, hidden_dim=256, action_space=None):
        super(GuassianPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, n_actions)

        self.log_std = nn.Linear(hidden_dim, n_actions)
        
        self.apply(weights_init)

        if action_space == None:
            self.action_range = 1.
            self.action_bias = 0.
        else:
            self.action_range = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        # We are also including a log_std, since we are learning a Gaussian distribution instead of a deterministic one
        # Log_std to limit the exploding values
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean * self.action_range + self.action_bias, log_std
    def sample(self, input_state):
        mean, log_std = self.forward(input_state)
        log_std = log_std.exp() # log of standard_deviation exponentiated 
        normal = Normal(mean, log_std) # guassian distribution generated
        x_t = normal.rsample() # reparemeterization trick, using rsample(), random variable u, with mew(u|s)
        # x_t is the equivalent to just doing a forward pass through the network, thus we need to put a tanh
        y_t = torch.tanh(x_t) # a = tanh(u)
        action = y_t * self.action_range + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.sum(torch.log(1 - y_t.pow(2)))
        mean = torch.tanh(mean) * self.action_range + self.action_bias # let's return the determinstic output as well
        return action, log_prob, mean

class QNetwork(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=256):
        super(QNetwork, self).__init__()
        
        # Two netwr
        self.linear1 = nn.Linear(n_states + n_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear4 = nn.Linear(n_states + n_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init)
    
    def forward(self, states, actions):
        states = torch.Tensor(states)
        actions = torch.Tensor(actions)
        x = torch.cat([states, actions], 1)
        # Two dueling network, take min of it
        x1 = F.relu(self.linear1(x))
        x1 = F.relu(self.linear2(x))
        x1 = self.linear3(x)

        x2 = F.relu(self.linear4(x))
        x2 = F.relu(self.linear5(x))
        x2 = self.linear6(x)

        return x1, x2
    
class ValueNetwork(nn.Module):
    def __init__(self, n_states, hidden_dim=256):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(n_states, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init)
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x  = F.relu(self.linear3(x))
        return x
   
# refactoring all the shit networks here into modular actor/critics

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
        x = torch.tensor(x)
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
        x = torch.tensor(x)
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
class ActorCritic(nn.Module):
    def __init__(self, n_actions=18, action_value=True):
        super(ActorCritic, self).__init__()
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
            return action, cat.log_prob(action), state_value[action], cat.entropy().mean()
        return action, cat.log_prob(action), state_value, cat.entropy().mean()


class PPOActorCritic(nn.Module):
    def __init__(self, num_actions=18):
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.Relu()
        )
        self.critic = nn.Linear(512, 1)
        self.actor = nn.Linear(512, num_actions)
    def forward(self, x): # x is multiple
        hidden = self.feature_extraction(x)
        critic = self.critic(hidden) #V(s)
        actor = self.actor(hidden)
        return actor, critic
    def get_batch_actions(self, states):
        actor, critic = self(states)
        # actor will be a tensor of num_envs * states, and we want to sum along the num_envs axis
        distribution = F.softmax(actor, dim=-1) # softmax dim -1 does it across the states for each envs
        cat = Categorical(distribution)
        action = cat.sample(len(states)) # because it's batched
        return action, cat.log_prob(action), critic, cat.entropy().mean()
