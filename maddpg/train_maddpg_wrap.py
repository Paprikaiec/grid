import argparse
import multiprocessing
import sys
from pathlib import Path
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

# env config
with open("../config.yaml") as f:
    config_dict = yaml.safe_load(f)
    # env config
env_config_dict = config_dict['environment']
data_path = Path("../envs/data/Climate_Zone_" + str(env_config_dict['climate_zone']))
buildings_states_actions = '../envs/data/buildings_state_action_space.json'

n_agents = env_config_dict['houses_per_node'] * 32
path = "../envs/data/" + str(n_agents) + "_agents/"

grid_config_dict = {
    "model_name": str(env_config_dict['houses_per_node'] * 32) + "agents",  # default 6*32 = 192
    "data_path": data_path,
    "climate_zone": env_config_dict['climate_zone'],
    "buildings_states_actions_file": buildings_states_actions,
    "hourly_timesteps": 4,
    "max_num_houses": None,
    "net_path": path + "case33.p",
    "agent_path": path + "agent_" + str(n_agents) + "_zone_" + str(env_config_dict['climate_zone']) + ".pickle",
    "episode_limit": env_config_dict["max_cycles"]
}
env = GridEnv(**grid_config_dict)

np.random.seed(1234)
th.manual_seed(1234)

wrap_number = env_config_dict['wrap_number']
n_agents = int(env.n_agents / wrap_number)
n_states = env.get_obs_size() * wrap_number # maybe obs in maddpg
n_actions = env.get_total_actions() * wrap_number
capacity = 750
batch_size = 32

n_episode = config_dict["train"]["epochs"]
max_steps = config_dict["environment"]["max_cycles"]
episodes_before_train = 3

maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity,
                episodes_before_train)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor

run = wandb.init(project="grid",
                 entity="wangyiwen",
                 config={**env_config_dict},
                 name=f"{env_config_dict['learn_policy']}" + f"{env.n_agents}")

for i_episode in range(n_episode):
    obs, _ = env.reset()
    obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    total_reward = 0.0
    rr = np.zeros((n_agents,))
    stat = {}
    for t in range(max_steps):

        obs = obs.type(FloatTensor)
        obs_wrap = th.reshape(obs, (n_agents, -1))
        action_wrap = maddpg.select_action(obs_wrap).data.cpu()
        action = th.reshape(action_wrap, (env.n_agents, -1))
        reward, done, info = env.step(action.numpy())
        obs_ = env.get_obs()

        reward = th.FloatTensor(reward).type(FloatTensor)
        reward_wrap = th.reshape(reward, (n_agents, -1)).sum(axis=1).squeeze()
        obs_ = np.stack(obs_)
        obs_ = th.from_numpy(obs_).float()
        if t != max_steps - 1:
            next_obs = obs_
            next_obs_wrap = np.reshape(next_obs, (n_agents, -1))
        else:
            next_obs = None
            next_obs_wrap = None

        total_reward += reward_wrap.sum()
        rr += reward_wrap.cpu().numpy()
        maddpg.memory.push(obs_wrap.data, action_wrap, next_obs_wrap, reward_wrap)
        obs = next_obs

        c_loss, a_loss = maddpg.update_policy()

        for k, v in info.items():
            if k in stat.keys():
                stat[k] += v
            else:
                stat[k] = v
    maddpg.episode_done += 1
    print('Episode: %d, reward = %f' % (i_episode, total_reward))
    reward_record.append(total_reward)

    #wandb log
    stat['avg_reward'] = total_reward
    for k, v in stat.items():
        wandb.log({k: v / max_steps}, step=i_episode)

run.finish()
