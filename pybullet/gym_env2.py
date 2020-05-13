from spinup import ppo_pytorch as ppo
import torch
import gym
from pybullet_envs.gym_locomotion_envs import HumanoidFlagrunBulletEnv

ac_kwargs = dict(hidden_sizes=[32,32], activation=torch.nn.ReLU)

logger_kwargs = dict(output_dir='/home/chk/文档/data', exp_name='HumanoidFlagrunBulletEnv')

ppo(env_fn=HumanoidFlagrunBulletEnv, ac_kwargs=ac_kwargs, 
    steps_per_epoch=1000, epochs=10, logger_kwargs=logger_kwargs)