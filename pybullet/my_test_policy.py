import gym
import time
from spinup.utils.test_policy import load_policy_and_env, run_policy

_, get_action = load_policy_and_env('/home/chk/文档/Centaur/data/ppo_humanoidflagrunbulletenv-v0/ppo_humanoidflagrunbulletenv-v0_s20')
env = gym.make("HumanoidFlagrunBulletEnv-v0", render=True)
run_policy(env, get_action)
