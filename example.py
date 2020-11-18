import gym
import time
import numpy as np
# import pybullet_envs
from spinup.utils.test_policy import load_policy_and_env, run_policy

env = gym.make("ChkDogBulletEnv-v0", render=True)
env.reset()
for _ in range(2000):
    env.reset()
    for _ in range(10000):
        # s, r, d, _ = env.step(np.zeros(env.action_space.sample().shape))
        s, r, d, _ = env.step(env.action_space.sample())
        if d:env.reset()
        time.sleep(1./240.)