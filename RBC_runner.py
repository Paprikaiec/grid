import csv
import os
import pickle

from agent.agents import MyAgents
from agent.rbc_agent import RBC_Agent_v2 as RBC_agent
from envs.smart_grid.smart_grid_env import GridEnv
from common.reply_buffer import *

import wandb


class RunnerRBC(object):

    def __init__(self, env: GridEnv):
        self.env = env
        self.train_config = ConfigObjectFactory.get_train_config()
        self.env_config = ConfigObjectFactory.get_environment_config()
        self.current_epoch = 0
        self.result_buffer = []
        self.env_info = self.env.get_env_info()

        # Warning: RBC_agent only give one building's actions
        self.agents = RBC_agent(self.env.buildings[self.env.agents[0]])

    def run_marl(self):
        run_episode = self.train_config.run_episode_before_train if "ppo" in self.env_config.learn_policy else 1
        for epoch in range(self.current_epoch, self.train_config.epochs + 1):
            total_reward = 0
            stat = {}
            for i in range(run_episode): # default: run_episode=3
                obs, _ = self.env.reset()
                finish_game = False
                cycle = 0
                while not finish_game and cycle < self.env_config.max_cycles:
                    state = self.env.get_state()
                    actions = [self.agents.predict()] * self.env.n_agents
                    rewards, finish_game, infos = self.env.step(actions)
                    obs_next = self.env.get_obs()
                    state_next = self.env.get_state()
                    total_reward += rewards
                    obs = obs_next
                    cycle += 1

                    # info: percentage of voltage out of control
                    #       voltage violtation
                    #       line loss
                    #       reactive power (q) loss
                    for k, v in infos.items():
                        if k in stat.keys():
                            stat[k] += v / run_episode
                        else:
                            stat[k] = v / run_episode


            # avg_reward = self.evaluate()
            avg_reward = total_reward / run_episode
            # wandb log
            stat['avg_reward'] = avg_reward
            for k, v in stat.items():
                wandb.log({k: v / cycle}, step=epoch)
            print("episode_{} over, avg_reward {}".format(epoch, avg_reward))
