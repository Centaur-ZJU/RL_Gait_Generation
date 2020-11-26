import os
import gym
import time
import numpy as np


env = gym.make('ChkCentaurEnv-v0', robot_name="yobo",render=True, precision_a=100, precision_s = 100)
s = env.reset()
print(len(s))
for _ in range(10000):
    # s, r, d, info = env.step(env.action_space.sample())
    # if d:
    #     print(s, info)
    #     env.reset()
    time.sleep(1./240.)

