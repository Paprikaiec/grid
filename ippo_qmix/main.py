import csv
import os

import matplotlib.pyplot as plt

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
import torch

from ..envs.smart_grid.smart_grid_env import GridEnv
from runner import RunnerSimpleSpreadEnv
from utils.config_utils import ConfigObjectFactory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('alg', type=str, nargs="?", default="ippo")
    argv = parser.parse_args()

    # env config
    climate_zone = 1
    data_path = Path("../envs/data/Climate_Zone_" + str(climate_zone))
    buildings_states_actions = '../envs/data/buildings_state_action_space.json'
    env_config_dict = {
        "model_name": "192agents",
        "data_path": data_path,
        "climate_zone": climate_zone,
        "buildings_states_actions_file": buildings_states_actions,
        "hourly_timesteps": 4,
        "max_num_houses": None
    }
    env = GridEnv(**env_config_dict)

    try:
        runner = RunnerSimpleSpreadEnv(env)
        runner.run_marl()
    finally:
        env.close()


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
    evaluate()
