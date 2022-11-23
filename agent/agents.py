import random

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical, MultivariateNormal

from policy.centralized_ppo import CentralizedPPO
from policy.independent_ppo import IndependentPPO
from policy.cuppo import CUPPO
from policy.qmix import QMix
from utils.config_utils import ConfigObjectFactory


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
        if self.env_config.learn_policy == "qmix":
            self.n_actions = self.env_info['n_actions']
            self.policy = QMix(self.env_info)

        elif self.env_config.learn_policy == "centralized_ppo":
            self.action_space = self.env_info['n_actions']
            self.policy = CentralizedPPO(self.env_info)

        elif self.env_config.learn_policy == "independent_ppo":
            self.action_space = self.env_info['action_space']
            self.policy = IndependentPPO(self.env_info)

        elif self.env_config.learn_policy == "cuppo":
            self.action_space = self.env_info['action_space']
            self.policy = CUPPO(self.env_info)

        else:
            raise ValueError(
                "learn_policy error, just support cuppo, "
                "qmix, centralized_ppo")

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
        if isinstance(self.policy, QMix):
            actions_ind = [i for i in range(self.n_actions)]
            for i, agent in enumerate(self.env_info['agents_name']):
                inputs = list()
                inputs.append(obs[i, :])
                inputs.append(torch.zeros(self.n_actions))
                agent_id = torch.zeros(self.n_agents)
                agent_id[i] = 1
                inputs.append(agent_id)
                inputs = torch.cat(inputs).unsqueeze(dim=0).to(self.device)
                with torch.no_grad():
                    hidden_state = self.policy.eval_hidden[:, i, :]
                    q_value, _ = self.policy.rnn_eval(inputs, hidden_state)
                if random.uniform(0, 1) > self.train_config.epsilon:
                    action = random.sample(actions_ind, 1)[0]
                else:
                    action = int(torch.argmax(q_value.squeeze()))
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
        elif isinstance(self.policy, IndependentPPO): # TODO: suitable for cuppo?
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
