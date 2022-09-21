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

from envs.smart_grid.smart_grid_env import GridEnv

from tensorboardX import SummaryWriter

MAX_ENV_STEPS=4*4*8759

# tic = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('alg', type=str, nargs="?", default="ippo")
argv = parser.parse_args()

#env config
climate_zone = 1
data_path = Path("./envs/data/Climate_Zone_"+str(climate_zone))
buildings_states_actions = './envs/data/buildings_state_action_space.json'
env_config_dict = {
    "model_name":"1024agents",
    "data_path":data_path,
    "climate_zone":climate_zone,
    "buildings_states_actions_file":buildings_states_actions,
    "hourly_timesteps":4,
    "max_num_houses":None
}
env = GridEnv(**env_config_dict)

#alg config
with open("./args/default.yaml", "r") as f:
	default_config_dict = yaml.safe_load(f)
with open("./args/alg_args/" + argv.alg + ".yaml", "r") as f:
	alg_config_dict = yaml.safe_load(f)["alg_args"]
	alg_config_dict["action_scale"] = 1.0
	alg_config_dict["action_bias"] = 0.0
alg_config_dict = {**default_config_dict, **alg_config_dict}
alg_config_dict["agent_num"] = env.get_num_of_agents()
alg_config_dict["obs_size"] = env.get_obs_size()
alg_config_dict["action_dim"] = env.get_total_actions()
args = convert(alg_config_dict)

# use wandb
run = wandb.init(project="grid",
				 entity="wangyiwen",
				 config={**env_config_dict, **alg_config_dict},
				 name=f"{argv.alg}" + f"{env.get_num_of_agents()}")

# create the logger(wandb)
logger = wandb
# logger = SummaryWriter("./model_save/" + "tensorboard/" + f"{argv.alg}")

model = Model[argv.alg]
print(f"{args}\n")
train = PGTrainer(args, model, env, logger)

for i in range(args.train_episodes_num):
	stat = {}
	train.run(stat, i)
	train.logging(stat)
	if i % args.save_model_freq == args.save_model_freq - 1:
		train.print_info(stat)
		torch.save({"model_state_dict": train.behaviour_net.state_dict()},
				"./model_save/" + f"{argv.alg}" + f"{env.get_num_of_agents()}" + "/model.pt")
		print("The model is saved!\n")

# logger.close()
run.finish()


# grid = GridLearn(**config)
#
# env = MyEnv(grid) #for _ in range(config['nclusters'])
# env.grid = grid
# env.initialize_rbc_agents()
#
# print("---env.grid.clusters---")
# print(env.grid.clusters)
#
# print("---env.grid.rl_agent---")
# print(env.grid.rl_agents)


# print('creating pettingzoo env...')
# env = ss.pettingzoo_env_to_vec_env_v0(env)
#
# print('stacking vec env...')
# # nenvs = 2
# env = ss.concat_vec_envs_v0(env, 1, num_cpus=1, base_class='stable_baselines3')
#
# grid.normalize_reward()
#
# model = PPO(MlpPolicy, env, ent_coef=0.1, learning_rate=0.001, n_epochs=30, verbose=2)
#
# # nloops=1
# # for loop in range(nloops):
# env.reset()
# print('==============')
# model.learn(4*4*8759)
# print('==============')
# if not os.path.exists(f'models/{model_name}'):
#     os.makedirs(f'models/{model_name}')
# os.chdir(f'models/{model_name}')
# # for m in range(len(models)):
# #     print('saving trained model')
# model.save(f"model")
# os.chdir('../..')
#
# toc = time.time()
# print(toc-tic)
# print('finished')
