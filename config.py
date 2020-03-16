#!/Users/zhenwang/software/anaconda3/envs/ai/bin/python
# -*- encoding: utf-8 -*-
"""
@File           :   config.py
@Time           :   2020_01_28-17:28:33
@Author         :   zhenwang
@Description    :
  - Version 1.0.0: File created.
"""
import torch

SEED = 1

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

STATE_TYPE_LIST = [
    "joint_positions",
    "joint_velocities",
    # 'gripper_pose',
    'gripper_joint_positions',
    'gripper_touch_forces',
    'gripper_open',
    # 'wrist_rgb',
    # 'wrist_mask',
    # 'left_shoulder_rgb',
    # 'right_shoulder_rgb',
    'task_low_dim_state',
]

HIDDEN_SIZE = 8

ACTOR_LR = 5e-4  # learning rate
ACTOR_HIDDEN_UNITS = (128, 64)
ACTOR_WEIGHT_DECAY = 1e-5

CRITIC_LR = 1e-3  # learning rate
CRITIC_HIDDEN_UNITS = (128, 64)
CRITIC_WEIGHT_DECAY = 1e-5

MEMORY_BUFFER_SIZE = int(1e6)  # maximum size of replay buffer
BATCH_SIZE = 64  # mini-batch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # interpolation parameter

N_EPISODES = 5000  # total episodes to train
EPS_START = 0.95  # initial value for exploration (epsilon)
EPS_DECAY = 2e-5  # epsilon decay value after each step
EPS_END = 0.05  # lower limit of epsilon
MAX_STEPS = 50  # maximum training steps of each epoch
LEARN_EVERY_STEP = BATCH_SIZE - 1  # extra learning after every step
