import gym
import time
from spinup.utils.test_policy import load_policy_and_env, run_policy

fpath = '/home/chk/文档/Centaur/'
fname = 'data/09-16-07_ppo_legs/09-16-07_ppo_legs_s9'
_, get_action = load_policy_and_env("".join([fpath, fname]))
env = gym.make("ChkHumanoidBulletEnv-v0", render=True)
run_policy(env, get_action)
