import pybullet
import gym, gym.spaces, gym.utils
import numpy as np
import os, inspect
import pybullet_data
import json
# import pybullet_data


class ChkRobot:
  def __init__(self, bullet_client, setting_file_path, robot_name):
    with open(setting_file_path, 'r') as f:
      robots_settings = json.load(f)
      if robot_name not in robots_settings.keys():
        raise ValueError("没有这个机器人：", robot_name)
      robot_settings = robots_settings[robot_name]
    self.robot_name = robot_settings["robot_name"]
    self.self_collision = robot_settings["self_collision"]
    self.action_dim = robot_settings["action_dim"]
    self.obs_dim = robot_settings["obs_dim"]
    self.motor_names = robot_settings["motor_names"]
    self.initial_z = robot_settings["initial_z"]
    self.foot_names = robot_settings["foot_names"]
    self._p = bullet_client
    high = np.ones([self.action_dim])
    self.action_space = gym.spaces.Box(-high, high)
    high = np.inf * np.ones([self.obs_dim])
    self.observation_space = gym.spaces.Box(-high, high)
    self.loadRobot(robot_settings["robot_file_path"])
    self.motors = [self.jdict[name] for name in self.motor_names]
    self.foot_ids = set(self.parts[f].bodyPartIndex for f in self.foot_names)

  def reset(self, bullet_client):
    self.robot_body.reset_position([0, 0, self.initial_z])
    if self.robot_name=='centaur':
      self.robot_body.reset_orientation(self._p.getQuaternionFromEuler([0, 0, -1.57]))
    return self.calc_state()

  def loadRobot(self, robot_file_path):
    self.parts = {}
    self.objects = []
    self.jdict = {}
    self.ordered_joints = []
    self.robot_body = None
    body = self._p.loadURDF(  robot_file_path,
                    flags=pybullet.URDF_USE_SELF_COLLISION | pybullet.URDF_GOOGLEY_UNDEFINED_COLORS)
    self.bodyId = body
    dump = 0
    for j in range(self._p.getNumJoints(body)):
        self._p.setJointMotorControl2(body, j, pybullet.POSITION_CONTROL,
                                      positionGain=0.1, velocityGain=0.1, force=0)
        jointInfo = self._p.getJointInfo(body, j)
        joint_name = jointInfo[1].decode("utf8")
        part_name = jointInfo[12].decode("utf8")

        if dump: print("ROBOT PART '%s'" % part_name)
        if dump:
          print(
              "ROBOT JOINT '%s'" % joint_name
          )  # limits = %+0.2f..%+0.2f effort=%0.3f speed=%0.3f" % ((joint_name,) + j.limits()) )

        self.parts[part_name] = BodyPart(self._p, part_name, body, j)
        # print(part_name, body, j)

        if part_name == self.robot_name:
          self.robot_body = self.parts[part_name]

        if j == 0 and self.robot_body is None:  # if nothing else works, we take this as robot_body
          self.parts[self.robot_name] = BodyPart(self._p, self.robot_name, body, -1)
          self.robot_body = self.parts[self.robot_name]


        if "ignore" in joint_name:
          Joint(self._p, joint_name, body, j).disable_motor()
          continue

        if 'fix' not in joint_name and "camera" not in joint_name:
          self.jdict[joint_name] = Joint(self._p, joint_name, body, j)
          self.ordered_joints.append(self.jdict[joint_name])

          self.jdict[joint_name].power_coef = 100.0

  def reset_pose(self, position, orientation):
    self.parts[self.robot_name].reset_pose(position, orientation)

  def apply_action(self, a):
    assert (np.isfinite(a).all())
    for n, j in enumerate(self.motors):
      limit = np.abs(j.upperLimit if a[n]>0 else j.lowerLimit)
      j.set_position(a[n] * limit)
    return {}

  def calc_state(self):
    j = np.array([j.current_relative_position() for j in self.ordered_joints],
                 dtype=np.float32).flatten()
    # even elements [0::2] position, scaled to -1..+1 between limits
    # odd elements  [1::2] angular speed, scaled to show -1..+1
    self.joint_speeds = j[1::2]
    self.torques = np.array([j.get_torque() for j in self.ordered_joints])

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
    more = np.array(
        [z, r, p, yaw],
        dtype=np.float32)

    return np.concatenate([more] + [j])


  def calc_potential(self):
    return self.robot_body.get_position()[0]

  def isFallDown(self):
    contact_ids = set(info[4] for info in self._p.getContactPoints(0))
    return True if (contact_ids - self.foot_ids) else False

  def alive(self):
    return -100 if self.isFallDown() else 1



class Pose_Helper:  # dummy class to comply to original interface

  def __init__(self, body_part):
    self.body_part = body_part

  def xyz(self):
    return self.body_part.current_position()

  def rpy(self):
    return pybullet.getEulerFromQuaternion(self.body_part.current_orientation())

  def orientation(self):
    return self.body_part.current_orientation()


class BodyPart:

  def __init__(self, bullet_client, body_name, body, bodyPartIndex):
    self.body = body
    self._p = bullet_client
    self.bodyPartIndex = bodyPartIndex
    self.initialPosition = self.current_position()
    self.initialOrientation = self.current_orientation()
    self.bp_pose = Pose_Helper(self)

  def state_fields_of_pose_of(
      self, body_id,
      link_id=-1):  # a method you will most probably need a lot to get pose and orientation
    if link_id == -1:
      (x, y, z), (a, b, c, d) = self._p.getBasePositionAndOrientation(body_id)
    else:
      (x, y, z), (a, b, c, d), _, _, _, _ = self._p.getLinkState(body_id, link_id)
    return np.array([x, y, z, a, b, c, d])

  def get_position(self):
    return self.current_position()

  def get_pose(self):
    return self.state_fields_of_pose_of(self.body, self.bodyPartIndex)

  def speed(self):
    if self.bodyPartIndex == -1:
      (vx, vy, vz), _ = self._p.getBaseVelocity(self.body)
    else:
      (x, y, z), (a, b, c, d), _, _, _, _, (vx, vy, vz), (vr, vp, vy) = self._p.getLinkState(
          self.body, self.bodyPartIndex, computeLinkVelocity=1)
    return np.array([vx, vy, vz])

  def current_position(self):
    return self.get_pose()[:3]

  def current_orientation(self):
    return self.get_pose()[3:]

  def get_orientation(self):
    return self.current_orientation()

  def reset_position(self, position):
    self._p.resetBasePositionAndOrientation(self.body, position,
                                            self.get_orientation())

  def apply_external_force(self, force):
    self._p.applyExternalForce(self.body, self.bodyPartIndex, force, 
                               self.current_position(), self._p.WORLD_FRAME)

  def reset_orientation(self, orientation):
    self._p.resetBasePositionAndOrientation(self.body, self.get_position(),
                                            orientation)

  def reset_velocity(self, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0]):
    self._p.resetBaseVelocity(self.body, linearVelocity, angularVelocity)

  def reset_pose(self, position, orientation):
    self._p.resetBasePositionAndOrientation(self.body, position, orientation)

  def pose(self):
    return self.bp_pose

  def contact_list(self):
    return self._p.getContactPoints(self.body, -1, self.bodyPartIndex, -1)


class Joint:

  def __init__(self, bullet_client, joint_name, body, jointIndex):
    self.body = body
    self._p = bullet_client
    self.jointIndex = jointIndex
    self.joint_name = joint_name

    jointInfo = self._p.getJointInfo(self.body, self.jointIndex)
    self.lowerLimit = jointInfo[8]
    self.upperLimit = jointInfo[9]

    self.power_coeff = 0

  def set_state(self, x, vx):
    self._p.resetJointState(self.body, self.jointIndex, x, vx)

  def current_position(self):  # just some synonyme method
    return self.get_state()

  def current_relative_position(self):
    pos, vel = self.get_state()
    pos_mid = 0.5 * (self.lowerLimit + self.upperLimit)
    return (2 * (pos - pos_mid) / (self.upperLimit - self.lowerLimit), 0.1 * vel)

  def get_state(self):
    x, vx, _, _ = self._p.getJointState(self.body, self.jointIndex)
    return x, vx

  def get_position(self):
    x, _ = self.get_state()
    return x

  def get_orientation(self):
    _, r = self.get_state()
    return r

  def get_velocity(self):
    _, vx = self.get_state()
    return vx

  def set_position(self, position, max_force=10):
    self._p.setJointMotorControl2(self.body,
                                  self.jointIndex,
                                  pybullet.POSITION_CONTROL,
                                  targetPosition=position,
                                  force=max_force)

  def set_velocity(self, velocity):
    self._p.setJointMotorControl2(self.body,
                                  self.jointIndex,
                                  pybullet.VELOCITY_CONTROL,
                                  targetVelocity=velocity)

  def set_motor_torque(self, torque):  # just some synonyme method
    self.set_torque(torque)

  def get_torque(self):
    _, _, _, torque = self._p.getJointState(self.body, self.jointIndex)
    return torque

  def set_torque(self, torque):
    self._p.setJointMotorControl2(bodyIndex=self.body,
                                  jointIndex=self.jointIndex,
                                  controlMode=pybullet.TORQUE_CONTROL,
                                  force=torque)  #, positionGain=0.1, velocityGain=0.1)

  def reset_current_position(self, position, velocity):  # just some synonyme method
    self.reset_position(position, velocity)

  def reset_position(self, position, velocity):
    self._p.resetJointState(self.body,
                            self.jointIndex,
                            targetValue=position,
                            targetVelocity=velocity)
    self.disable_motor()

  def disable_motor(self):
    self._p.setJointMotorControl2(self.body,
                                  self.jointIndex,
                                  controlMode=pybullet.POSITION_CONTROL,
                                  targetPosition=0,
                                  targetVelocity=0,
                                  positionGain=0.1,
                                  velocityGain=0.1,
                                  force=0)