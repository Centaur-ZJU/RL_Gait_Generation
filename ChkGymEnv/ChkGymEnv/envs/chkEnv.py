import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import pybullet
import time
import os
from ChkGymEnv.envs.chkRobot import ChkRobot
from pybullet_utils import bullet_client
from pkg_resources import parse_version

try:
  if os.environ["PYBULLET_EGL"]:
    import pkgutil
except: pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

robot_setting_file_path = os.path.join(BASE_DIR, 'robot_settings/setting.json')
plane_stadium_path = os.path.join(BASE_DIR, 'data/plane.urdf')


class ChkCentaurEnv(gym.Env):
  def __init__(self, robot_name="yobo", render=False, debug=False):
    self.physicsClientId = -1
    self.ownsPhysicsClient = 0
    self.isRender = render
    self.buildPhysicsClient()
    self.loadScene()
    self.robot = ChkRobot(self._p, robot_setting_file_path, robot_name)
    self.seed()
    self.action_space = self.robot.action_space
    self.observation_space = self.robot.observation_space
    self.stateId = -1
    self.reset()
    self.fixedTimeStep = self._p.getPhysicsEngineParameters()['fixedTimeStep']
    self.debug_mode = debug
    

  def loadScene(self):
    self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
    self.planeId = self._p.loadURDF(plane_stadium_path)
    self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

  def buildPhysicsClient(self):
    if (self.physicsClientId < 0):
      self.ownsPhysicsClient = True

      if self.isRender:
        self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
      else:
        self._p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
      self._p.resetSimulation()
      self._p.setGravity(0, 0, -9.8)
      self._p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
      #optionally enable EGL for faster headless rendering
      try:
        if os.environ["PYBULLET_EGL"]:
          con_mode = self._p.getConnectionInfo()['connectionMethod']
          if con_mode==self._p.DIRECT:
            egl = pkgutil.get_loader('eglRenderer')
            if (egl):
              self._p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
            else:
              self._p.loadPlugin("eglRendererPlugin")
      except:
        pass
      self.physicsClientId = self._p._client
      self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

  def seed(self, seed=None):
    pass

  def reset(self):
    self.steps = 0
    self.done = 0
    self.reward = 0
    s = self.robot.reset(self._p)
    self.potential = self.robot.calc_potential()
    if (self.stateId >= 0):
      #print("restoreState self.stateId:",self.stateId)
      self._p.restoreState(self.stateId)
    if (self.stateId < 0):
      self.stateId = self._p.saveState()
      #print("saving state self.stateId:",self.stateId)
    return s

  def _isDone(self):
    return self.robot.isFallDown()

  def close(self):
    if (self.ownsPhysicsClient):
      if (self.physicsClientId >= 0):
        self._p.disconnect()
    self.physicsClientId = -1

  progress_weight = 1.
  electricity_cost_weight = -.01
  alive_weight = 1.

  def render(self):
    pass

  def step(self, a):
    old_potential = self.robot.calc_potential()
    info = self.robot.apply_action(a)
    self._p.stepSimulation()
    self.steps += 1
    if self.isRender:
      time.sleep(1./240.)
    robot_state = self.robot.calc_state()
    self._alive = self.robot.alive()
    done = self._isDone()
    state = robot_state

    if not np.isfinite(state).all():
      print("~INF~", state)
      done = True
    
    # 奖励设计
    ## 前进奖励
    progress = float(self.robot.calc_potential()-old_potential) / self.fixedTimeStep
    progress_reward = self.progress_weight * progress

    ## 存活奖励
    alive_reward = self.alive_weight * self._alive

    ## 功率惩罚
    electricity_joints = np.abs(self.robot.torques * self.robot.joint_speeds)
    electricity_cost = self.electricity_cost_weight * float(electricity_joints.mean())

    step_reward = progress_reward + alive_reward + electricity_cost


    if self.debug_mode:
      print("robot position:",self.robot.robot_body.get_position())
      print("progress_reward:", progress_reward)
      print("alive_reward:", alive_reward)
      print("electricity_cost:", electricity_cost)
      print("step_reward:",step_reward)

    self.reward += step_reward
    return state, step_reward, bool(done), info


