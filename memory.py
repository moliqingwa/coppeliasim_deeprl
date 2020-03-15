# -*- encoding: utf-8 -*-
"""
@File           :   memory.py
@Time           :   2020_01_26-22:00:26
@Author         :   zhenwang
@Description    :
  - Version 1.0.0: File created.
"""
import random
from collections import deque, namedtuple

import numpy as np
import torch

from config import DEVICE


class ReplyBuffer(object):
    """
    Fixed-size buffer to store experience tuples.
    """
    def __init__(self, buffer_size, seed):
        """
        Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from memory.
        Params
        ======
            batch_size (int): size of each training batch
        """
        experiences = random.sample(self.memory, k=batch_size)

        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
