# coding: utf-8

from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt

from ddpg_agent import Agent
from config import *
from rlbench_env import RLBenchEnv

action_mode = ActionMode(ArmActionMode.ABS_JOINT_TORQUE)
env = RLBenchEnv("ReachTarget",
                 state_type_list=[
                     "joint_positions",
                     "joint_velocities",
                     'gripper_pose',
                     # 'wrist_rgb',
                     # 'wrist_mask',
                     # 'left_shoulder_rgb',
                     # 'right_shoulder_rgb',
                     'task_low_dim_state',
                 ])
state = env.reset()
action_dim = env.action_space.shape[0]
state_space = env.observation_space

agent = Agent(state_space, HIDDEN_SIZE, action_dim, 1,
              seed=SEED, buffer_size=MEMORY_BUFFER_SIZE,
              actor_lr=ACTOR_LR, actor_hidden_sizes=ACTOR_HIDDEN_UNITS, actor_weight_decay=ACTOR_WEIGHT_DECAY,
              critic_lr=CRITIC_LR, critic_hidden_sizes=CRITIC_HIDDEN_UNITS, critic_weight_decay=CRITIC_WEIGHT_DECAY,
              batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU
              )
print(agent)
agent.load()

scores_deque = deque(maxlen=100)
scores, actor_losses, critic_losses = [], [], []
for i_episode in range(1, 100):
    state = env.reset()
    agent.reset()

    avg_score, rewards = 0, []
    actor_loss_list, critic_loss_list = [], []
    for t in range(MAX_STEPS):
        actions = agent.act(state, add_noise=True)

        action = actions.ravel()
        next_state, reward, terminate, _ = env.step(action)
        done = 1. if terminate else 0.

        rewards.append(reward)
        state = next_state
        if done:
            break
    avg_score = np.mean(rewards) if rewards else 0.
    scores_deque.append(avg_score)
    scores.append(avg_score)
    actor_losses.append(np.mean(actor_loss_list) if actor_loss_list else 0.)
    critic_losses.append(np.mean(critic_loss_list) if critic_loss_list else 0.)
    print(f"\rEpisode {i_episode}\t"
          f"Average Score: {np.mean(scores_deque):.2f}\tCurrent Score: {avg_score:.2f}\t"
          f"Actor Loss: {np.mean(actor_loss_list) if actor_loss_list else 0:.2e}"
          f"\tCritic Loss: {np.mean(critic_loss_list) if critic_loss_list else 0.:.2e}")

    if i_episode % 100 == 0:
        # agent.save()
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    if i_episode % 50 == 0:
        agent.save()
        print('Save Model\n\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    if np.mean(scores_deque) >= 0:
        print("Scores are >= 0, quit")
        agent.save()
        break


fig = plt.figure()
ax1 = fig.add_subplot(311)
ax1.plot(np.arange(1, len(scores) + 1), scores)
ax1.set_ylabel('Score')
ax1.set_xlabel('Episode #')

ax2 = fig.add_subplot(312)
ax2.plot(np.arange(1, len(actor_losses) + 1), actor_losses)
# ax2.legend()
ax2.set_ylabel('Actor Loss')
ax2.set_xlabel('Episode #')

ax3 = fig.add_subplot(313)
ax3.plot(np.arange(1, len(critic_losses) + 1), critic_losses)

ax3.set_ylabel('Critic Loss')
ax3.set_xlabel('Episode #')
plt.savefig("training.png")

print('Done')
env.close()
