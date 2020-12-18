import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import pybullet
import time
import os
from ChkGymEnv.envs.chkRobot import ChkRobot
from pybullet_utils import bullet_client
from pkg_resources import parse_version
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

try:
  if os.environ["PYBULLET_EGL"]:
    import pkgutil
except: pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

robot_setting_file_path = os.path.join(BASE_DIR, 'robot_settings/setting.json')
plane_stadium_path = os.path.join(BASE_DIR, 'data/plane.urdf')


class ChkCentaurEnv(gym.Env):
  def __init__(self, robot_name="centaur", render=False, debug=False, precision_s=0, precision_a=0,
                dynamic_weight=False, fps=20, max_force=20):
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
    self.dynamic_weight = dynamic_weight
    self.max_force=max_force
    # if precision_s!=0 or precision_a!=0 :
    #   self.adjustSpace()
    self.stateId = -1
    self.reset()
    self.fixedTimeStep = 1./fps
    self._p.setPhysicsEngineParameter(fixedTimeStep=self.fixedTimeStep)
    self.debug_mode = debug
    self.foot_stand = np.array([0] * len(self.robot.foot_names))
    self.stand_steps = np.array([0] * len(self.robot.foot_names))
    self.maxFoot_height = np.array([0.] * len(self.robot.foot_names), dtype=np.float)
    self.writer = SummaryWriter('policy_test')
    
  def log(self):
    result = {}
    result['fl_hip_angle']=self.robot.jdict['flhip'].get_position()
    result['fr_hip_angle']=self.robot.jdict['frhip'].get_position()
    result['bl_hip_angle']=self.robot.jdict['blhip'].get_position()
    result['br_hip_angle']=self.robot.jdict['brhip'].get_position()
    result['fl_knee_angle']=self.robot.jdict['flknee'].get_position()
    result['fr_knee_angle']=self.robot.jdict['frknee'].get_position()
    result['bl_knee_angle']=self.robot.jdict['blknee'].get_position()
    result['br_knee_angle']=self.robot.jdict['brknee'].get_position()
    result['height']=self.robot.robot_body.get_position()[2]
    result['torques']=self.robot.torques.mean()
    result['electricity']=np.abs(self.robot.torques * self.robot.joint_speeds).mean()
    for k in result.keys():
      self.writer.add_scalar('centaur/'+k, result[k], global_step=self.steps)

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
    self.foot_stand = np.array([0] * len(self.robot.foot_names))
    self.stand_steps = np.array([0] * len(self.robot.foot_names))
    self.maxFoot_height = np.array([0.] * len(self.robot.foot_names), dtype=np.float)
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
  hangFoot_weight = .5


  def render(self):
    pass

  def env_state(self, robot_state):
    if self.precision_s!=0:
      env_state = (robot_state*self.precision_s).astype(np.int) *1. / self.precision_s
    else:
      env_state = robot_state
    
    contact_ids = set(info[4] for info in self._p.getContactPoints(self.planeId))
    for i, f in enumerate(self.robot.foot_names):
      self.foot_stand[i] = self.robot.parts[f].bodyPartIndex in contact_ids
    env_state = np.concatenate((env_state, self.foot_stand))
    return env_state

  def step(self, a):
    # self.log()
    old_potential = self.robot.calc_potential()
    if self.precision_a!=0:
      info = self.robot.apply_action((a * self.precision_a).astype(np.int) * 1. / self.precision_a,
                                      self.max_force)
    else:
      info = self.robot.apply_action(a,self.max_force)
    self._p.stepSimulation()
    self.steps += 1
    if self.isRender:
      self._p.resetDebugVisualizerCamera(3, 30, -30, self.robot.robot_body.get_position())
      time.sleep(self.fixedTimeStep)
    robot_state = self.robot.calc_state()
    self._alive = self.robot.alive()
    done = self._alive<0
    old_foot_stand = self.foot_stand.copy()
    env_state = self.env_state(robot_state)
    
        

    if not np.isfinite(robot_state).all():
      print("~INF~", robot_state)
      done = True
    
    if self.dynamic_weight:
      clip_ratio = np.clip((self.subRewards["progress_r"])/100., 0.1, 1)
      # print("dynamic_weight is active")
    else:
      clip_ratio = 1
    # 奖励设计
    ## 前进奖励
    progress = float(self.robot.calc_potential()-old_potential) / self.fixedTimeStep
    progress_reward = self.progress_weight * progress
    progress /= clip_ratio

    ## 存活奖励
    alive_reward = self.alive_weight * self._alive

    ## 功率惩罚
    electricity_joints = np.abs(self.robot.torques * self.robot.joint_speeds)
    _electricity = float(electricity_joints.mean())
    electricity_cost = self.electricity_cost_weight * _electricity
    electricity_cost *= clip_ratio

    ## 姿态惩罚
    body_pose = self._p.getEulerFromQuaternion(self.robot.robot_body.get_orientation())
    _pose = (body_pose[0]**2 + body_pose[1]**2)
    pose_cost = self.pose_weight * _pose
    pose_cost *= clip_ratio

    ## 足部悬空高度奖励
    hangFoot_reward = 0
    for i in range(len(old_foot_stand)):
      foot_name = self.robot.foot_names[i]
      self.maxFoot_height[i] = max(self.robot.parts[foot_name].get_position()[2],self.maxFoot_height[i])
      if old_foot_stand[i]==0 and self.foot_stand[i]==True:
        hangFoot_reward += 1. * (self.steps-self.stand_steps[i]) * self.maxFoot_height[i]
        self.stand_steps[i] = self.steps
        self.maxFoot_height[i] = 0.
    hangFoot_reward *= self.hangFoot_weight
    hangFoot_reward *= clip_ratio
    # print(self.maxFoot_height)
        

    subRewards = {"alive_r":alive_reward, "electricity_c":electricity_cost, 
                  "progress_r":progress_reward, "pose_c":pose_cost}#, "hangHeight_r":hangFoot_reward}

    subScores = {"alive":self._alive, "electricity":_electricity, 
                  "progress":progress, "pose":_pose}
    
    step_reward = sum(subRewards.values())

    if self.debug_mode:
      print("robot position:",self.robot.robot_body.get_position())
      print(subRewards)
      print("step_reward:",step_reward)

    
    self.reward += step_reward
    for k in subRewards.keys():
      self.subRewards[k] += subRewards[k]
    info = dict(subScores, **info)
    return env_state, step_reward, bool(done), info


