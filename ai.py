# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn # for Neural Network
import torch.nn.functional as F # for some functions that we need
import torch.optim as optim # for some optimizer to perform stochastic gradient descent
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network

class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30) # Fully Connections, We have 30 hidden Nodes in hidden layers.
        self.fc2 = nn.Linear(30, nb_action)

    def forward(self, state):

        # activate hidden neurns
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# Implementing Experience Replay

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))       
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
