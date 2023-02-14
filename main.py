import csv
import os

import matplotlib.pyplot as plt

import argparse
import multiprocessing
import sys
from pathlib import Path
import numpy as np
import random
import time
import yaml
import wandb
import torch

from envs.smart_grid.smart_grid_env import GridEnv
from runner import *
from utils.config_utils import ConfigObjectFactory


def main():
    with open("./config.yaml") as f:
        config_dict = yaml.safe_load(f)
    # env config
    env_config_dict = config_dict['environment']
    data_path = Path("./envs/data/Climate_Zone_" + str(env_config_dict['climate_zone']))
    buildings_states_actions = './envs/data/buildings_state_action_space.json'

    n_agents = env_config_dict['houses_per_node'] * 32
    path = "./envs/data/" + str(n_agents) + "_agents/"

    grid_config_dict = {
        "model_name": str(env_config_dict['houses_per_node']*32)+"agents", # default 6*32 = 192
        "data_path": data_path,
        "climate_zone": env_config_dict['climate_zone'],
        "buildings_states_actions_file": buildings_states_actions,
        "hourly_timesteps": 4,
        "max_num_houses": None,
        "houses_per_node": env_config_dict['houses_per_node'],
        "net_path": path + "case33.p",
        "agent_path": path + "agent_" + str(n_agents) + "_zone_" + str(env_config_dict['climate_zone']) + ".pickle"
    }
    env = GridEnv(**grid_config_dict)

    maxday = env_config_dict["max_cycles"] / 96
    run = wandb.init(project="grid",
                     entity="wangyiwen",
                     config={**env_config_dict},
                     name=f"{env_config_dict['learn_policy']}" + f"_zone_{env_config_dict['climate_zone']}")
                     # f"{env_config_dict['learn_policy']}" + f"_{env.n_agents}" + f"_maxday_{maxday}")

    if env_config_dict["learn_policy"] == "RBC":
        runner = RunnerRBC(env)
    elif "evaluate" in env_config_dict["mode"]:
        runner = Runnerevaluate(env)
    else:
        runner = RunnerGrid(env)

    runner.run_marl()

    run.finish()

def evaluate():
    train_config = ConfigObjectFactory.get_train_config()
    env_config = ConfigObjectFactory.get_environment_config()
    csv_filename = os.path.join(train_config.result_dir, env_config.learn_policy, "result.csv")
    rewards = []
    total_rewards = []
    avg_rewards = 0
    len_csv = 0
    with open(csv_filename, 'r') as f:
        r_csv = csv.reader(f)
        for data in r_csv:
            total_rewards.append(round(float(data[0]), 2))
            avg_rewards += float(data[0])
            if len_csv % train_config.show_evaluate_epoch == 0 and len_csv > 0:
                rewards.append(round(avg_rewards / train_config.show_evaluate_epoch, 2))
                avg_rewards = 0
            len_csv += 1

    plt.plot([i * train_config.show_evaluate_epoch for i in range(len_csv // train_config.show_evaluate_epoch)],
             rewards)
    plt.plot([i for i in range(len_csv)], total_rewards, alpha=0.3)
    plt.title("rewards")
    plt.show()


if __name__ == "__main__":
    main()
