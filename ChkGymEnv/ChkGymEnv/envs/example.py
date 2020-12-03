import os
import gym
import time
import numpy as np


env = gym.make('ChkCentaurEnv-v0', robot_name="centaur",render=True, precision_a=100, precision_s = 100)
s = env.reset()
print(len(s))
for _ in range(10000):
    s, r, d, info = env.step(env.action_space.sample())
    print(s, r, d, info)
    time.sleep(1./240.)

