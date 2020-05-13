import gym
import logging
import random
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding


logger = logging.getLogger(__name__)

class CentaurEnv(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self):
        self.viewer = None
        # 定义动作空间和状态空间，spinup 正常训练必须要有这两货
        # 动作空间是2维的，浮点数，取值范围是 [-1,1] 
        # 状态空间是3维的，浮点数，取值范围是 [-1,1]
        self.action_space = spaces.Box(low=np.array([-1,-1]), 
                                        high=np.array([1,1]), 
                                        shape=None,dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        s1, s2 = self.state
        s1 = np.random.random_sample()*action[1]
        s2 = np.random.random_sample()*action[0]
        self.state = np.array([s1, s2])
        costs = (s1-s2)**2
        done = costs<1e-10
        return self._get_obs(), -costs, done, {}

    def _get_obs(self):
        s1, s2 = self.state
        return np.array([s1, s2, s1*s2])

    def reset(self):
        high = np.array([1,1])
        self.state = self.np_random.uniform(low=-high, high=high)
        return self._get_obs()

    def render(self, mode='human'):
        print("render")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None