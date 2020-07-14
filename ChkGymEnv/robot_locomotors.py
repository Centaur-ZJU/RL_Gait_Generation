from robot_bases import XmlBasedRobot, MJCFBasedRobot, URDFBasedRobot
import numpy as np
import pybullet
import os
# import pybullet_data
from robot_bases import BodyPart


class WalkerBase(MJCFBasedRobot):

  def __init__(self, fn, robot_name, action_dim, obs_dim, power):
    MJCFBasedRobot.__init__(self, fn, robot_name, action_dim, obs_dim)
    self.power = power
    self.camera_x = 0
    self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
    self.walk_target_x = 1e3  # kilometer away
    self.walk_target_y = 0
    self.body_xyz = [0, 0, 0]

  def robot_specific_reset(self, bullet_client):
    self._p = bullet_client
    for j in self.ordered_joints:
      j.reset_current_position(self.np_random.uniform(low=-0.1, high=0.1), 0)

    self.feet = [self.parts[f] for f in self.foot_list]
    self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)
    self.scene.actor_introduce(self)
    self.initial_z = None

  def apply_action(self, a):
    assert (np.isfinite(a).all())
    for n, j in enumerate(self.ordered_joints):
      j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))

  def calc_state(self):
    j = np.array([j.current_relative_position() for j in self.ordered_joints],
                 dtype=np.float32).flatten()
    # even elements [0::2] position, scaled to -1..+1 between limits
    # odd elements  [1::2] angular speed, scaled to show -1..+1
    self.joint_speeds = j[1::2]
    self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

    body_pose = self.robot_body.pose()
    parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
    self.body_xyz = (parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2]
                    )  # torso z is more informative than mean z
    self.body_real_xyz = body_pose.xyz()
    self.body_rpy = body_pose.rpy()
    z = self.body_xyz[2]
    if self.initial_z == None:
      self.initial_z = z
    r, p, yaw = self.body_rpy
    self.walk_target_theta = np.arctan2(self.walk_target_y - self.body_xyz[1],
                                        self.walk_target_x - self.body_xyz[0])
    self.walk_target_dist = np.linalg.norm(
        [self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0]])
    angle_to_target = self.walk_target_theta - yaw

    rot_speed = np.array([[np.cos(-yaw), -np.sin(-yaw), 0], [np.sin(-yaw),
                                                             np.cos(-yaw), 0], [0, 0, 1]])
    vx, vy, vz = np.dot(rot_speed,
                        self.robot_body.speed())  # rotate speed back to body point of view

    more = np.array(
        [
            z - self.initial_z,
            np.sin(angle_to_target),
            np.cos(angle_to_target),
            0.3 * vx,
            0.3 * vy,
            0.3 * vz,  # 0.3 is just scaling typical speed into -1..+1, no physical sense here
            r,
            p
        ],
        dtype=np.float32)
    return np.clip(np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)

  def calc_potential(self):
    # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
    # all rewards have rew/frame units and close to 1.0
    debugmode = 0
    if (debugmode):
      print("calc_potential: self.walk_target_dist")
      print(self.walk_target_dist)
      print("self.scene.dt")
      print(self.scene.dt)
      print("self.scene.frame_skip")
      print(self.scene.frame_skip)
      print("self.scene.timestep")
      print(self.scene.timestep)
    return -self.walk_target_dist / self.scene.dt

class ChkWalkerBase(MJCFBasedRobot):

  def __init__(self, fn, robot_name, action_dim, obs_dim, power):
    MJCFBasedRobot.__init__(self, fn, robot_name, action_dim, obs_dim)
    self.power = power
    self.camera_x = 0
    self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
    self.walk_target_x = 1e3  # kilometer away
    self.walk_target_y = 0
    self.body_xyz = [0, 0, 0]

  def robot_specific_reset(self, bullet_client):
    self._p = bullet_client
    for j in self.ordered_joints:
      j.reset_current_position(self.np_random.uniform(low=-0.1, high=0.1), 0)

    self.feet = [self.parts[f] for f in self.foot_list]
    self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)
    self.scene.actor_introduce(self)
    self.initial_z = None

  def apply_action(self, a):
    assert (np.isfinite(a).all())
    for n, j in enumerate(self.ordered_joints):
      j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))

  def calc_state(self):
    j = np.array([j.current_relative_position() for j in self.ordered_joints],
                 dtype=np.float32).flatten()
    # even elements [0::2] position, scaled to -1..+1 between limits
    # odd elements  [1::2] angular speed, scaled to show -1..+1
    self.joint_speeds = j[1::2]
    self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

    body_pose = self.robot_body.pose()
    parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
    self.body_xyz = (parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2]
                    )  # torso z is more informative than mean z
    self.body_real_xyz = body_pose.xyz()
    self.body_rpy = body_pose.rpy()
    z = self.body_xyz[2]
    if self.initial_z == None:
      self.initial_z = z
    r, p, yaw = self.body_rpy
    self.walk_target_theta = np.arctan2(self.walk_target_y - self.body_xyz[1],
                                        self.walk_target_x - self.body_xyz[0])
    self.walk_target_dist = np.linalg.norm(
        [self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0]])
    angle_to_target = self.walk_target_theta - yaw

    rot_speed = np.array([[np.cos(-yaw), -np.sin(-yaw), 0], [np.sin(-yaw),
                                                             np.cos(-yaw), 0], [0, 0, 1]])
    vx, vy, vz = np.dot(rot_speed,
                        self.robot_body.speed())  # rotate speed back to body point of view

    more = np.array(
        [
            z - self.initial_z,
            np.sin(angle_to_target),
            np.cos(angle_to_target),
            0.3 * vx,
            0.3 * vy,
            0.3 * vz,  # 0.3 is just scaling typical speed into -1..+1, no physical sense here
            r,
            p
        ],
        dtype=np.float32)
    return np.clip(np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)

  def calc_potential(self):
    # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
    # all rewards have rew/frame units and close to 1.0
    debugmode = 0
    if (debugmode):
      print("calc_potential: self.walk_target_dist")
      print(self.walk_target_dist)
      print("self.scene.dt")
      print(self.scene.dt)
      print("self.scene.frame_skip")
      print(self.scene.frame_skip)
      print("self.scene.timestep")
      print(self.scene.timestep)
    return -self.walk_target_dist / self.scene.dt



class ChkHumanoid(ChkWalkerBase):
  self_collision = True
  foot_list = ["right_foot", "left_foot"]  # "left_hand", "right_hand"
  action_dim=11
  obs_dim=10+action_dim*2    # 10 + action_dim * 2
  def __init__(self):
    WalkerBase.__init__(self,
                        'chk_humanoid_symmetric.xml',
                        'torso',
                        action_dim=self.action_dim,
                        obs_dim=self.obs_dim,
                        power=0.41)
    # 17 joints, 4 of them important for walking (hip, knee), others may as well be turned off, 17/4 = 4.25

  def robot_specific_reset(self, bullet_client):
    WalkerBase.robot_specific_reset(self, bullet_client)
    self.motor_names = ["abdomen_x"]
    self.motor_power = [100]
    self.motor_names += ["right_hip_x", "right_hip_z", "right_hip_y", "right_knee"]
    self.motor_power += [200, 200, 300, 200]
    self.motor_names += ["left_hip_x", "left_hip_z", "left_hip_y", "left_knee"]
    self.motor_power += [200, 200, 300, 200]
    self.motor_names += ["left_ankle", "right_ankle"]
    self.motor_power += [200, 200]
    self.motors = [self.jdict[n] for n in self.motor_names]
    self.tired = np.zeros((self.action_dim,))    # 每个关节的疲劳程度
    if self.random_yaw:
      position = [0, 0, 0]
      orientation = [0, 0, 0]
      yaw = self.np_random.uniform(low=-3.14, high=3.14)
      if self.random_lean and self.np_random.randint(2) == 0:
        cpose.set_xyz(0, 0, 1.4)
        if self.np_random.randint(2) == 0:
          pitch = np.pi / 2
          position = [0, 0, 0.45]
        else:
          pitch = np.pi * 3 / 2
          position = [0, 0, 0.25]
        roll = 0
        orientation = [roll, pitch, yaw]
      else:
        position = [0, 0, 1.4]
        orientation = [0, 0, yaw]  # just face random direction, but stay straight otherwise
      self.robot_body.reset_position(position)
      self.robot_body.reset_orientation(orientation)
    self.initial_z = 0.8
    if self.random_pose:
      random_range = 0.1
      position = [0, 0, 1.2]
      orientation = np.array([0, 0, 0])+np.random.random((3,))*random_range*2-random_range
      for m in self.motors:
        motor_position = np.random.random()*random_range*2-random_range
        motor_velocity = np.random.random()*random_range*2-random_range
        m.reset_position(2*motor_position, motor_velocity)   # 关节初始化范围
      self.robot_body.reset_position(position)  # 位置初始化范围,；初始高度太小会飞起来，因为将脚的位置设在了地板以下，导致初始时地板的作用力很大
      self.robot_body.reset_orientation(self._p.getQuaternionFromEuler(2*orientation))
      self.initial_z = position[2]
    

  random_yaw = False
  random_lean = False
  random_pose = True
  external_force = 20
  joint_damp = False

  def apply_action(self, a):
    assert (np.isfinite(a).all())
    force_gain = 1
    a = np.clip(a, -1, +1)
    damping = 1-np.clip(self.tired-0.2, 0, 0.1)/0.1
    for i, m, power in zip(range(self.action_dim), self.motors, self.motor_power):
      m.set_motor_torque(float(force_gain * power * self.power * (a[i] * damping[i])))
    if self.joint_damp:
      self.tired = 0.999 * self.tired + 0.001 * (a>0.5)
    if self.external_force>0:
      force = np.random.normal([0, 0.1, 3])*self.external_force
      force[2] = 0
      self.robot_body.apply_external_force(force)
    # print(self.tired[-1], damping[-1])
    # for name in self.motor_names:
    #   if name=='left_hip_x':
    #     self.jdict[name].set_motor_torque(-1)
    #   elif name == 'right_hip_y':
    #     self.jdict[name].set_motor_torque(-1)
    #   else:
    #     self.jdict[name].set_motor_torque(0)
    # self.robot_body.reset_position(np.array([0, 0, 1.2]))
    # self.robot_body.reset_orientation([0, 0, 0, 1])
    # self.jdict['left_hip_z'].reset_position(-60*0.0175, 0)
    # self.jdict['right_hip_y'].reset_position(120*0.0175, 0)

  def alive_bonus(self, z, pitch):
    return +1.5 if z > 0.78 else -1  # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying

class Humanoid(WalkerBase):
  self_collision = True
  foot_list = ["right_foot", "left_foot"]  # "left_hand", "right_hand"

  def __init__(self):
    WalkerBase.__init__(self,
                        'humanoid_symmetric.xml',
                        'torso',
                        action_dim=17,
                        obs_dim=44,
                        power=0.41)
    # 17 joints, 4 of them important for walking (hip, knee), others may as well be turned off, 17/4 = 4.25

  def robot_specific_reset(self, bullet_client):
    WalkerBase.robot_specific_reset(self, bullet_client)
    self.motor_names = ["abdomen_z", "abdomen_y", "abdomen_x"]
    self.motor_power = [100, 100, 100]
    self.motor_names += ["right_hip_x", "right_hip_z", "right_hip_y", "right_knee"]
    self.motor_power += [100, 100, 300, 200]
    self.motor_names += ["left_hip_x", "left_hip_z", "left_hip_y", "left_knee"]
    self.motor_power += [100, 100, 300, 200]
    self.motor_names += ["right_shoulder1", "right_shoulder2", "right_elbow"]
    self.motor_power += [75, 75, 75]
    self.motor_names += ["left_shoulder1", "left_shoulder2", "left_elbow"]
    self.motor_power += [75, 75, 75]
    self.motors = [self.jdict[n] for n in self.motor_names]
    if self.random_yaw:
      position = [0, 0, 0]
      orientation = [0, 0, 0]
      yaw = self.np_random.uniform(low=-3.14, high=3.14)
      if self.random_lean and self.np_random.randint(2) == 0:
        cpose.set_xyz(0, 0, 1.4)
        if self.np_random.randint(2) == 0:
          pitch = np.pi / 2
          position = [0, 0, 0.45]
        else:
          pitch = np.pi * 3 / 2
          position = [0, 0, 0.25]
        roll = 0
        orientation = [roll, pitch, yaw]
      else:
        position = [0, 0, 1.4]
        orientation = [0, 0, yaw]  # just face random direction, but stay straight otherwise
      self.robot_body.reset_position(position)
      self.robot_body.reset_orientation(orientation)
    self.initial_z = 0.8

  random_yaw = False
  random_lean = False

  def apply_action(self, a):
    assert (np.isfinite(a).all())
    force_gain = 1
    for i, m, power in zip(range(17), self.motors, self.motor_power):
      m.set_motor_torque(float(force_gain * power * self.power * np.clip(a[i], -1, +1)))

  def alive_bonus(self, z, pitch):
    return +2 if z > 0.78 else -1  # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying


