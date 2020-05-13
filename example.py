import gym
import time

env = gym.make('Centaur-v0')
env.reset()
env.render()
time.sleep(3)
env.close()