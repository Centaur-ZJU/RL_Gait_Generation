import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import pybullet
import time
import os
from ChkGymEnv.envs.chkRobot import ChkRobot
from pybullet_utils import bullet_client
from pkg_resources import parse_version
from collections import defaultdict

try:
  if os.environ["PYBULLET_EGL"]:
    import pkgutil
except: pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

robot_setting_file_path = os.path.join(BASE_DIR, 'robot_settings/setting.json')
plane_stadium_path = os.path.join(BASE_DIR, 'data/plane.urdf')


class ChkCentaurEnv(gym.Env):
  def __init__(self, robot_name="yobo", render=False, debug=False, precision_s=0, precision_a=0):
    self.physicsClientId = -1
    self.ownsPhysicsClient = 0
    self.isRender = render
    self.buildPhysicsClient()
    self.loadScene()
    self.robot = ChkRobot(self._p, robot_setting_file_path, robot_name)
    self.seed()
    self.action_space = self.robot.action_space
    self.observation_space = self.robot.observation_space
    self.precision_s, self.precision_a = precision_s, precision_a
    # if precision_s!=0 or precision_a!=0 :
    #   self.adjustSpace()
    self.stateId = -1
    self.reset()
    self.fixedTimeStep = 1./20.
    self._p.setPhysicsEngineParameter(fixedTimeStep=self.fixedTimeStep)
    self.debug_mode = debug
    
  def adjustSpace(self):
    if self.precision_a!=0:
      self.action_space = gym.spaces.Box(0-self.precision_a, self.precision_a, 
                                        shape=self.action_space.shape, dtype=int)
    if self.precision_s!=0:
      self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=self.observation_space.shape, 
                                                dtype=int)


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

  def seed(self, seed=0):
    pass

  def reset(self):
    self.steps = 0
    self.done = 0
    self.reward = 0
    self.subRewards = defaultdict(float)
    robot_s = self.robot.reset(self._p)
    self.potential = self.robot.calc_potential()
    if (self.stateId >= 0):
      #print("restoreState self.stateId:",self.stateId)
      self._p.restoreState(self.stateId)
    if (self.stateId < 0):
      self.stateId = self._p.saveState()
      #print("saving state self.stateId:",self.stateId)
    return self.env_state(robot_s)

  def _isDone(self):
    return self.robot.isFallDown()

  def close(self):
    if (self.ownsPhysicsClient):
      if (self.physicsClientId >= 0):
        self._p.disconnect()
    self.physicsClientId = -1

  progress_weight = 3.
  electricity_cost_weight = -.1
  alive_weight = .1
  pose_weight = -1.

  def render(self):
    pass

  def env_state(self, robot_state):
    if self.precision_s!=0:
      env_state = (robot_state*self.precision_s).astype(np.int) *1. / self.precision_s
    else:
      env_state = robot_state
    foot_stand = np.array([0] * len(self.robot.foot_names))
    contact_ids = set(info[4] for info in self._p.getContactPoints(self.planeId))
    for i, f in enumerate(self.robot.foot_names):
      foot_stand[i] = self.robot.parts[f].bodyPartIndex in contact_ids
    env_state = np.concatenate((env_state, foot_stand))
    return env_state

  def step(self, a):
    old_potential = self.robot.calc_potential()
    if self.precision_a!=0:
      info = self.robot.apply_action((a * self.precision_a).astype(np.int) * 1. / self.precision_a)
    else:
      info = self.robot.apply_action(a)
    self._p.stepSimulation()
    self.steps += 1
    if self.isRender:
      self._p.resetDebugVisualizerCamera(3, 30, -30, self.robot.robot_body.get_position())
      time.sleep(self.fixedTimeStep)
    robot_state = self.robot.calc_state()
    self._alive = self.robot.alive()
    done = self._alive<0

    if not np.isfinite(robot_state).all():
      print("~INF~", robot_state)
      done = True
    
    clip_ratio = np.clip((self.subRewards["progress_r"])/100., 0.1, 1)
    # 奖励设计
    ## 前进奖励
    progress = float(self.robot.calc_potential()-old_potential) / self.fixedTimeStep
    progress_reward = self.progress_weight * progress
    progress /= clip_ratio

    ## 存活奖励
    alive_reward = self.alive_weight * self._alive

    ## 功率惩罚
    electricity_joints = np.abs(self.robot.torques * self.robot.joint_speeds)
    electricity_cost = self.electricity_cost_weight * float(electricity_joints.mean())
    electricity_cost *= clip_ratio

    ## 姿态惩罚
    body_pose = self._p.getEulerFromQuaternion(self.robot.robot_body.get_orientation())
    pose_cost = self.pose_weight * (body_pose[0]**2 + body_pose[1]**2)
    pose_cost *= clip_ratio

    subRewards = {"alive_r":alive_reward, "electricity_c":electricity_cost, 
                  "progress_r":progress_reward, "pose_c":pose_cost}
    
    step_reward = sum(subRewards.values())

    if self.debug_mode:
      print("robot position:",self.robot.robot_body.get_position())
      print(subRewards)
      print("step_reward:",step_reward)

    
    self.reward += step_reward
    for k in subRewards.keys():
      self.subRewards[k] += subRewards[k]
    info = dict(subRewards, **info)
    return self.env_state(robot_state), step_reward, bool(done), info


