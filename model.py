# -*- encoding: utf-8 -*-
"""
@File           :   model.py
@Time           :   2020_01_26-15:34:36
@Author         :   zhenwang
@Description    :
  - Version 1.0.0: File created.
"""
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DEVICE


def layer_init(layer, w_scale=1.0):
    # nn.init.orthogonal_(layer.weight.data)
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    layer.weight.data.uniform_(-lim, lim)
    layer.weight.data.mul_(w_scale)

    nn.init.constant_(layer.bias.data, 0)
    return layer


class Backbones(nn.Module):
    """ Extract features from different types of states

        for vector state, dim(shape)==1: MLP
        for tensor (image) state, dim(shape)>1: CNN
    """

    def __init__(self, state_space, hidden_size):
        super(Backbones, self).__init__()
        self.device = DEVICE
        self.hidden = nn.ModuleList()  # need to use ModuleList for various number of layers on cuda
        self.cnn_tail = nn.ModuleList()
        self.after_conv_size_list = []

        for single_state_space in state_space:
            state_dim = single_state_space.shape[0]
            if len(single_state_space.shape) == 1:  # vector
                self.hidden.append(nn.Sequential(
                    nn.Linear(state_dim, hidden_size),
                    nn.ReLU(True)
                ))
            elif len(single_state_space.shape) > 1:  # tensor, images, etc
                assert single_state_space.shape[0] == single_state_space.shape[1]  # square image
                input_channel = 3
                output_channel = 32
                after_conv_size = int(
                    output_channel * (state_dim / 4) ** 2)  # calculate the dimension after convolution
                self.after_conv_size_list.append(after_conv_size)

                self.hidden.append(nn.Sequential(
                    nn.Conv2d(input_channel, int(output_channel / 2), kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    nn.ReLU(True),
                    nn.BatchNorm2d(int(output_channel / 2)),

                    nn.Conv2d(int(output_channel / 2), output_channel, kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    nn.ReLU(True),
                    nn.BatchNorm2d(output_channel),
                ))

                self.cnn_tail.append(nn.Linear(after_conv_size, hidden_size))
            else:
                raise ValueError('Wrong State Shape!')

    def forward(self, state_list):
        x = []
        idx = 0
        for (state, layer) in zip(np.rollaxis(state_list, 1), self.hidden):  # first dim is N: number of samples
            # list of array to array
            if len(state[0].shape) > 1:
                state = np.vstack([state.tolist()])  # vstack lose first dimension for more than 1 dim tensor
                state = torch.FloatTensor(state).to(self.device)
                z = layer(state)
                z = z.view(-1, self.after_conv_size_list[idx])
                x.append(self.cnn_tail[idx](z))
                idx += 1
            else:
                state = np.vstack(state)
                state = torch.FloatTensor(state).to(self.device)
                x.append(layer(state))
        output = torch.cat(x, dim=-1)

        return output


class Actor(nn.Module):
    def __init__(self, state_space, hidden_size, action_size,
                 seed=0,
                 hidden_units=(400, 300)):
        """
        Initialize parameters and build the actor model.
        Params
        ======
            state_space (tuple): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_units (tuple): Dimensions of sequence hidden layers
        """
        super().__init__()
        self.state_space = state_space
        self.hidden_size = hidden_size
        self.action_size = action_size

        self.seed = torch.manual_seed(seed)

        if not hidden_units or not len(hidden_units):
            raise Exception(f"hidden_units({hidden_units}) should NOT be empty!")

        self.backbones = Backbones(state_space, hidden_size)
        sum_hidden_dim = len(state_space) * hidden_size

        hidden_gate_func = nn.LeakyReLU

        layers = []
        previous_features = sum_hidden_dim
        for idx, hidden_size in enumerate(hidden_units):
            layers.append(layer_init(nn.Linear(previous_features, hidden_size)))
            # layers.append(nn.BatchNorm1d(hidden_size))  # adding batch norm
            layers.append(hidden_gate_func(inplace=True))

            previous_features = hidden_size

        layers.append(layer_init(nn.Linear(previous_features, action_size), 3e-3))
        layers.append(nn.Tanh())
        self.fc_body = nn.Sequential(*layers)

    def forward(self, state):
        x = self.backbones(state)
        return self.fc_body(x)


class Critic(nn.Module):
    def __init__(self, state_space, hidden_size, action_size,
                 seed=0,
                 hidden_units=(400, 300)):
        """
        Initialize parameters and build the critic model.
        Params
        =======
            state_space (tuple): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_units (tuple): Dimensions of sequence hidden layers
        """
        super().__init__()
        self.state_space = state_space
        self.hidden_size = hidden_size
        self.action_size = action_size

        self.seed = torch.manual_seed(seed)

        if not hidden_units or not len(hidden_units):
            raise Exception(f"hidden_units({hidden_units}) should NOT be empty!")

        self.backbones = Backbones(state_space, hidden_size)
        sum_hidden_dim = len(state_space) * hidden_size

        hidden_gate_func = nn.LeakyReLU

        self.fc_body = nn.Sequential(
            nn.Linear(sum_hidden_dim, hidden_units[0]),
            hidden_gate_func(inplace=True),
        )

        layers = []
        previous_features = hidden_units[0] + action_size
        for hidden_size in hidden_units[1:]:
            layers.append(layer_init(nn.Linear(previous_features, hidden_size)))
            # layers.append(nn.BatchNorm1d(hidden_size))  # adding batch norm
            layers.append(hidden_gate_func(inplace=True))

            previous_features = hidden_size

        layers.append(layer_init(nn.Linear(previous_features, 1), 3e-3))
        # layers.append(nn.ReLU(inplace=True))  # using ReLU, because the value should NOT be negative.
        self.critic_body = nn.Sequential(*layers)

    def forward(self, state, action):
        x = self.backbones(state)
        x = self.fc_body(x)
        x = torch.cat((x, action), dim=1)
        return self.critic_body(x)

