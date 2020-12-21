import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution

from utils import ReplayBuffer, TanhTransform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLPNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, layers=[256, 256, 256], activation=nn.functional.relu, lr=3e-4):
        super(MLPNetwork, self).__init__()

        self.net_layers = nn.ModuleList()

        for i in range(len(layers)):
            if i == 0:
                layer = nn.Linear(obs_dim + action_dim, layers[0])
                nn.init.xavier_uniform_(layer.weight.data)
            else:
                layer = nn.Linear(layers[i - 1], layers[i])
                nn.init.xavier_uniform_(layer.weight.data)

            self.net_layers.append(layer)

        layer = nn.Linear(layers[-1], 1)
        nn.init.xavier_uniform_(layer.weight.data)
        self.net_layers.append(layer)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.activation = activation

    def forward(self, data):
        for i in range(len(self.net_layers)-1):
            data = self.net_layers[i](data)
            data = self.activation(data)
        data = self.net_layers[-1](data)

        return data


class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim, layers=[256, 256], activation=nn.functional.relu, lr=3e-4):
        super(Policy, self).__init__()

        self.net_layers = nn.ModuleList()

        for i in range(len(layers)):
            if i == 0:
                layer = nn.Linear(obs_dim, layers[0])
                nn.init.xavier_uniform_(layer.weight.data)
            else:
                layer = nn.Linear(layers[i - 1], layers[i])
                nn.init.xavier_uniform_(layer.weight.data)

            self.net_layers.append(layer)

        self.mu_layer = nn.Linear(layers[-1], action_dim)
        self.logsigma_layer = nn.Linear(layers[-1], action_dim)
        nn.init.xavier_uniform_(self.mu_layer.weight.data)
        nn.init.xavier_uniform_(self.logsigma_layer.weight.data)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.activation = activation

    def forward(self, state, get_logprob=False):
        data = state
        for i in range(len(self.net_layers)):
            data = self.net_layers[i](data)
            data = self.activation(data)

        mu = self.mu_layer(data)
        logsigma = self.logsigma_layer(data)
        logstd = torch.clamp(logsigma, -20, 2)
        std = logstd.exp()
        dist = Normal(mu, std)
        transforms = [TanhTransform(cache_size=1)]
        dist = TransformedDistribution(dist, transforms)
        action = dist.rsample()
        logprob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        mean = torch.tanh(mu)
        return action, logprob, mean


class DoubleQFunc(nn.Module):

    def __init__(self, state_dim, action_dim, lr=3e-4):
        super(DoubleQFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim, action_dim)
        self.network2 = MLPNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)