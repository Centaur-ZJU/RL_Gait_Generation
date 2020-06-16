import gym
import time
# import pybullet_envs
from spinup.utils.test_policy import load_policy_and_env, run_policy

env = gym.make("HumanoidBulletEnv-v0", render=True)
env.reset()
for _ in range(10000):
    env.step(env.action_space.sample())
    time.sleep(.1)