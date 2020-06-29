import gym
import time
from spinup.utils.test_policy import load_policy_and_env, run_policy

fpath = '/home/chk/文档/Centaur/'
fname = 'data/2020-06-29_ppo_centaur-human/2020-06-29_ppo_centaur-human_s3'
_, get_action = load_policy_and_env("".join([fpath, fname]))
env = gym.make("HumanoidBulletEnv-v0", render=True)
run_policy(env, get_action)
