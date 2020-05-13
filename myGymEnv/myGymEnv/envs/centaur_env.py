import logging
import random

import gym
import numpy as np
import pybullet
import pybullet_data
from gym import error, spaces, utils
from gym.utils import seeding
from pybullet_utils import bullet_client

logger = logging.getLogger(__name__)

class CentaurEnv(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self, render=False):
        self.isRender = render
        self.scene = None
        # 定义动作空间和状态空间，spinup 正常训练必须要有这两货
        # 动作空间是2维的，浮点数，取值范围是 [-1,1] 
        # 状态空间是3维的，浮点数，取值范围是 [-1,1]
        self.action_space = spaces.Box(low=np.array([-1,-1]), 
                                        high=np.array([1,1]), 
                                        shape=None,dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.seed()
        self.physicsClientId = -1   # 连接的引擎id,为负数表示没有连接引擎
        self.ownsPhysicsClient = 0  # 是否有连接引擎


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
        # print(self.physicsClientId)
        if (self.physicsClientId<0):
            self.ownsPhysicsClient=True

            if self.isRender:
                self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            else:
                self._p = bullet_client.BulletClient()
            self._p.resetSimulation()
            self._p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
            self.physicsClientId = self._p._client
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeId = pybullet.loadURDF("plane.urdf")
        return self._get_obs()

    def render(self, mode='human'):
        self._p.stepSimulation()

    def close(self):
        pass
