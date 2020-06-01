import gym
import time
from spinup.utils.test_policy import load_policy_and_env, run_policy

fpath = '/home/chk/文档/Centaur/'
fname = 'data/2020-05-28_ppo_centaur-human_z=0-7/2020-05-28_ppo_centaur-human_z=0-7_s6'
_, get_action = load_policy_and_env("".join([fpath, fname]))
env = gym.make("HumanoidBulletEnv-v0", render=True)
run_policy(env, get_action)
