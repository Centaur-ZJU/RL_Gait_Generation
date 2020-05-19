import gym
import time
from spinup.utils.test_policy import load_policy_and_env, run_policy

fpath = '/home/chk/文档/Centaur/'
fname = 'data/2020-05-17_ppo_humanoidbulletenv-v0/2020-05-17_ppo_humanoidbulletenv-v0_s21'
_, get_action = load_policy_and_env("".join([fpath, fname]), deterministic=True)
env = gym.make("HumanoidBulletEnv-v0", render=True)
run_policy(env, get_action)