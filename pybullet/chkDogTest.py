import gym
import time
from spinup.utils.test_policy import load_policy_and_env, run_policy

fpath = '/home/chk/文档/Centaur/'
fname = 'data/14-11:20_ppo_four_legs/14-11:20_ppo_four_legs_s0'
_, get_action = load_policy_and_env("".join([fpath, fname]))
env = gym.make("ChkDogBulletEnv-v0", render=True)
run_policy(env, get_action)
