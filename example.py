import gym
import time
# import pybullet_envs
from spinup.utils.test_policy import load_policy_and_env, run_policy

# _, get_action = load_policy_and_env('/home/chk/文档/Centaur/data/ppo_humanoidflagrunbulletenv-v0_hid64-32_tanh/ppo_humanoidflagrunbulletenv-v0_hid64-32_tanh_s20')
env = gym.make("HopperBulletEnv-v0", render=True)
print(env.observation_space)
print(env.action_space)
env.reset()
for _ in range(1000):
    env.render()
    time.sleep(0.01)
    nextState, r, done, _ = env.step(env.action_space.sample())
    if done:env.reset()