import argparse
import multiprocessing
import sys
from pathlib import Path
import gym
import numpy as np
import random
import time
import yaml
import wandb
import torch as th

from envs.smart_grid.smart_grid_env import GridEnv
from MADDPG import MADDPG
from params import scale_reward

parser = argparse.ArgumentParser()
parser.add_argument('alg', type=str, nargs="?", default="ippo")
argv = parser.parse_args()

reward_record = []

#env config
climate_zone = 1
data_path = Path("./envs/data/Climate_Zone_"+str(climate_zone))
buildings_states_actions = './envs/data/buildings_state_action_space.json'
env_config_dict = {
    "model_name":"192agents",
    "data_path":data_path,
    "climate_zone":climate_zone,
    "buildings_states_actions_file":buildings_states_actions,
    "hourly_timesteps":4,
    "max_num_houses":None
}
env = GridEnv(**env_config_dict)

np.random.seed(1234)
th.manual_seed(1234)

n_agents = env.get_num_of_agents()
n_states = env.get_obs_size() # maybe obs in maddpg
n_actions = env.get_total_actions()
capacity = 1500
batch_size = 32

n_episode = 300
max_steps = 96
episodes_before_train = 5

maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity,
                episodes_before_train)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
for i_episode in range(n_episode):
    obs, _ = env.reset()
    obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    total_reward = 0.0
    rr = np.zeros((n_agents,))
    for t in range(max_steps):

        if t == 20:
            print('success 20 steps')
        obs = obs.type(FloatTensor)
        action = maddpg.select_action(obs).data.cpu()
        reward, done, _ = env.step(action.numpy())
        obs_ = env.get_obs()

        reward = th.FloatTensor(reward).type(FloatTensor)
        obs_ = np.stack(obs_)
        obs_ = th.from_numpy(obs_).float()
        if t != max_steps - 1:
            next_obs = obs_
        else:
            next_obs = None

        total_reward += reward.sum()
        rr += reward.cpu().numpy()
        maddpg.memory.push(obs.data, action, next_obs, reward)
        obs = next_obs

        c_loss, a_loss = maddpg.update_policy()
    maddpg.episode_done += 1
    print('Episode: %d, reward = %f' % (i_episode, total_reward))
    reward_record.append(total_reward)

    # Save model
    if t % 30 == 0:
        th.save({f'Actor{i}': MADDPG.actors[i] for i in range(n_agents)}, "./maddpg_model/Actor.tar")
        th.save({f'Critic{i}': MADDPG.critics[i] for i in range(n_agents)}, "./maddpg_model/Critic.tar")
        print('Model saved!')


