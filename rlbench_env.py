# coding: utf-8

import numpy as np
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
from gym import spaces

# list of state types
state_types = [
    'left_shoulder_rgb',
    'left_shoulder_depth',
    'left_shoulder_mask',
    'right_shoulder_rgb',
    'right_shoulder_depth',
    'right_shoulder_mask',
    'wrist_rgb',
    'wrist_depth',
    'wrist_mask',
    'joint_velocities',
    'joint_velocities_noise',
    'joint_positions',
    'joint_positions_noise',
    'joint_forces',
    'joint_forces_noise',
    'gripper_pose',
    'gripper_touch_forces',
    'task_low_dim_state',
]

image_types = [
    'left_shoulder_rgb',
    'left_shoulder_depth',
    'left_shoulder_mask',
    'right_shoulder_rgb',
    'right_shoulder_depth',
    'right_shoulder_mask',
    'wrist_rgb',
    'wrist_depth',
    'wrist_mask',
]

primitive_types = [
    'gripper_open',
]


class RLBenchEnv(object):
    """ make RLBench env to have same interfaces as openai.gym """

    def __init__(self, task_name, state_type_list=None, headless=False):
        if state_type_list is None:
            state_type_list = ['left_shoulder_rgb']
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
        self.env = Environment(
            action_mode, obs_config=obs_config, headless=headless)
        self.env.launch()
        try:
            self.task = self.env.get_task(ReachTarget)
        except:
            raise NotImplementedError

        _, obs = self.task.reset()

        if len(state_type_list) > 0:
            self.observation_space = []
            for state_type in state_type_list:
                state = getattr(obs, state_type)
                if isinstance(state, (int, float)):
                    shape = (1,)
                else:
                    shape = state.shape
                self.observation_space.append(spaces.Box(low=-np.inf, high=np.inf, shape=shape))
        else:
            raise ValueError('No State Type!')
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_mode.action_size,), dtype=np.float32)

        self.state_type_list = state_type_list

    def seed(self, seed_value):
        # set seed as in openai.gym env
        pass

    def render(self):
        # render the scene
        pass

    def _get_state(self, obs):
        if len(self.state_type_list) > 0:
            state = []
            for state_type in self.state_type_list:
                if state_type in image_types:
                    image = getattr(obs, state_type)
                    image = np.moveaxis(image, 2, 0)  # change (H, W, C) to (C, H, W) for torch
                    state.append(image)
                elif state_type in primitive_types:
                    state.append(np.array([getattr(obs, state_type)]))
                else:
                    state.append(getattr(obs, state_type))
        else:
            raise ValueError('State Type Not Exists!')
        return state

    def reset(self):
        descriptions, obs = self.task.reset()
        return self._get_state(obs)

    def step(self, action):
        obs, reward, terminate = self.task.step(action)  # reward in original rlbench is binary for success or not

        # distance between current pos (tip) and target pos
        # x, y, z, qx, qy, qz, qw = self.task._robot.arm.get_tip().get_pose()
        x, y, z, qx, qy, qz, qw = self.task._robot.gripper.get_pose()
        tar_x, tar_y, tar_z, _, _, _, _ = self.task._task.target.get_pose()
        distance = (np.abs(x - tar_x) + np.abs(y - tar_y) + np.abs(z - tar_z))
        reward -= 1. * distance

        # speed regulation
        joint_velocities = self.task._robot.arm.get_joint_velocities()
        velocities = np.inner(joint_velocities, joint_velocities)
        reward -= 0.05 / len(joint_velocities) * velocities

        # orientation
        gripper_x, gripper_y, gripper_z, gripper_qx, gripper_qy, gripper_qz, gripper_qw = self.task._robot.gripper.get_pose()
        v1 = np.array((tar_x - gripper_x, tar_y - gripper_y, tar_z - gripper_z))
        temp = np.sin(np.arccos(gripper_qw))
        v2 = np.array((gripper_qx / temp, gripper_qy / temp, gripper_qz / temp))
        orientation_angle = np.abs(np.arccos(v1.dot(v2) / np.sqrt(v1.dot(v1)) / np.sqrt(v2.dot(v2))))  # [0, pi]
        orientation_angle = orientation_angle if orientation_angle < np.pi / 2. else (np.pi - orientation_angle)
        reward -= 0.5 / np.pi * orientation_angle  # if orientation_angle > np.pi / 8. else 0.
        return self._get_state(obs), reward, terminate, None

    def close(self):
        self.env.shutdown()
