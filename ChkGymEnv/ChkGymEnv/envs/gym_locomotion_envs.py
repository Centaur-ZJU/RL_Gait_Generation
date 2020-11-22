import numpy as np
import pybullet
from chkEnv_bases import ChkRobotEnv
from chkRobot_dog import ChkDog


class chkDogEnv(ChkRobotEnv):
  def __init__(self, robot = None, render=False):
    if robot is None:
      self.robot = ChkDog()
    else:
      self.robot = robot
    ChkRobotEnv.__init__(self, self.robot, render)