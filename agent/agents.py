import random

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical, MultivariateNormal

from policy.centralized_ppo import CentralizedPPO
from policy.independent_ppo import IndependentPPO
from policy.cmaqmix import CMAQMix
from utils.config_utils import ConfigObjectFactory

from cmaes import CMA

class MyAgents:
    def __init__(self, env_info: dict):
        self.env_info = env_info
        self.train_config = ConfigObjectFactory.get_train_config()
        self.env_config = ConfigObjectFactory.get_environment_config()
        self.n_agents = self.env_info['n_agents']

        if self.train_config.cuda:
            torch.cuda.empty_cache()
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        # 初始化学习策略，需要注意的是不同算法对应不通的动作空间(连续/离散)
        # 下面三个算法作为baseline
        if "cmaqmix" in self.env_config.learn_policy:
            self.action_dim = self.env_info['action_dim']
            self.action_space = self.env_info['action_space']
            self.obs_shape = self.env_info['obs_shape']
            self.policy = CMAQMix(self.env_info)

        elif "centralized_ppo" in self.env_config.learn_policy:
            self.action_space = self.env_info['n_actions']
            self.policy = CentralizedPPO(self.env_info)

        elif "independent_ppo" in self.env_config.learn_policy:
            self.action_space = self.env_info['action_space']
            self.policy = IndependentPPO(self.env_info)
        else:
            raise ValueError(
                "learn_policy error, just support independent_ppo"
                "cmaqmix, centralized_ppo")

    def learn(self, batch_data: dict, episode_num: int = 0):
        self.policy.learn(batch_data, episode_num)

    def choose_actions(self, obs) -> tuple:
        actions_with_name = {}
        actions = []
        log_probs = []

        # smart grid obs is numpy ndarray
        # obs = torch.stack([torch.Tensor(value) for value in obs.values()], dim=0)
        obs = torch.Tensor(obs)
        self.policy.init_hidden(1)
        if isinstance(self.policy, CMAQMix):
            actions_ind = [i for i in range(self.action_dim)]
            for i, agent in enumerate(self.env_info['agents_name']):
                if (self.env_config.mode == "train") and random.uniform(0, 1) > self.train_config.epsilon:
                    action = self.action_space.sample() # todo:
                else: # todo implement CMA-ES choose action below
                    inputs = list()
                    inputs.append(obs[i, :])
                    inputs.append(torch.zeros(self.action_dim))
                    agent_id = torch.zeros(self.n_agents)
                    agent_id[i] = 1
                    inputs.append(agent_id)
                    inputs = torch.cat(inputs).unsqueeze(dim=0).to(self.device) #inputs dimension[1,obs+aciton+agent_id]
                    hidden_state = self.policy.eval_hidden[:, i, :]

                    # cmaes optimizer
                    optimizer = CMA(mean=np.zeros(self.action_dim), sigma=1.3)
                    for generation in range(50):
                        solutions = []
                        for _ in range(optimizer.population_size):
                            x = optimizer.ask()
                            inputs[0, self.obs_shape:self.obs_shape+self.action_dim] = torch.from_numpy(x)
                            with torch.no_grad():
                                q_value, _ = self.policy.rnn_eval(inputs, hidden_state)
                            solutions.append((x, - q_value.squeeze()))
                        optimizer.tell(solutions)

                    # find the minimum action w.r.t - q_values
                    value = 999999
                    for element in solutions:
                        if element[1] < value:
                            action = element[0]
                            value = element[1]

                actions_with_name[agent] = action
                actions.append(action)
        elif isinstance(self.policy, CentralizedPPO):
            obs = obs.reshape(1, -1).to(self.device)
            with torch.no_grad():
                action_means, _ = self.policy.ppo_actor(obs, self.policy.rnn_hidden)
            for i, agent_name in enumerate(self.env_info['agents_name']):
                action_mean = action_means[:, i].squeeze()
                dist = MultivariateNormal(action_mean, self.policy.get_cov_mat())
                action = np.clip(dist.sample().cpu().numpy(), self.action_space.low,
                                 self.action_space.high).astype(dtype=np.float32)
                log_probs.append(dist.log_prob(torch.Tensor(action).to(self.device)))
                actions_with_name[agent_name] = action
                actions.append(action)
        elif isinstance(self.policy, IndependentPPO):
            obs = obs.to(self.device)
            for i, agent_name in enumerate(self.env_info['agents_name']):
                with torch.no_grad():
                    action_mean, _ = self.policy.ppo_actor(obs[i].unsqueeze(dim=0), self.policy.rnn_hidden[i])
                action_mean = action_mean.squeeze()
                dist = MultivariateNormal(action_mean, self.policy.get_cov_mat())
                action = np.clip(dist.sample().cpu().numpy(), self.action_space.low,
                                 self.action_space.high).astype(dtype=np.float32)
                log_probs.append(dist.log_prob(torch.Tensor(action).to(self.device)))
                actions_with_name[agent_name] = action
                actions.append(action)
        return actions_with_name, actions, log_probs

    def save_model(self):
        self.policy.save_model()

    def load_model(self):
        self.policy.load_model()

    def del_model(self):
        self.policy.del_model()

    def is_saved_model(self) -> bool:
        return self.policy.is_saved_model()

    def get_results_path(self):
        return self.policy.result_path
