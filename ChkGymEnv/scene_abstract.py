import sys, os
sys.path.append(os.path.dirname(__file__))
import pybullet

import gym


class Scene:
  "A base class for single- and multiplayer scenes"

  def __init__(self, bullet_client, gravity, timestep, frame_skip):
    self._p = bullet_client     # 连接引擎
    self.np_random, seed = gym.utils.seeding.np_random(None)    # 随机种子
    self.timestep = timestep    # 设置步长
    self.frame_skip = frame_skip    # 跳帧

    self.dt = self.timestep * self.frame_skip   # 速度？
    self.cpp_world = World(self._p, gravity, timestep, frame_skip)    # 创建引擎空间

    self.test_window_still_open = True  # or never opened
    self.human_render_detected = False  # if user wants render("human"), we open test window

    self.multiplayer_robots = {}

  def test_window(self):
    "Call this function every frame, to see what's going on. Not necessary in learning."
    self.human_render_detected = True
    return self.test_window_still_open

  def actor_introduce(self, robot):
    "Usually after scene reset"
    if not self.multiplayer: return
    self.multiplayer_robots[robot.player_n] = robot

  def actor_is_active(self, robot):
    """
        Used by robots to see if they are free to exclusiveley put their HUD on the test window.
        Later can be used for click-focus robots.
        """
    return not self.multiplayer

  def episode_restart(self, bullet_client):
    "This function gets overridden by specific scene, to reset specific objects into their start positions"
    self.cpp_world.clean_everything()   # 清空引擎空间
    #self.cpp_world.test_window_history_reset()

  def global_step(self):
    """
        The idea is: apply motor torques for all robots, then call global_step(), then collect
        observations from robots using step() with the same action.
        """
    self.cpp_world.step(self.frame_skip)    # 迭代


class SingleRobotEmptyScene(Scene):
  multiplayer = False  # this class is used "as is" for InvertedPendulum, Reacher


class World:

  def __init__(self, bullet_client, gravity, timestep, frame_skip):
    self._p = bullet_client       # 连接引擎
    self.gravity = gravity      # 设置重力
    self.timestep = timestep    # 设置步长
    self.frame_skip = frame_skip    # 跳帧
    self.numSolverIterations = 5    # ？
    self.clean_everything()   # 清空引擎空间

  def clean_everything(self):
    #p.resetSimulation()
    self._p.setGravity(0, 0, -self.gravity)     # 重力重置
    self._p.setDefaultContactERP(0.9)       # ？
    #print("self.numSolverIterations=",self.numSolverIterations)
    self._p.setPhysicsEngineParameter(fixedTimeStep=self.timestep * self.frame_skip,    # 
                                      numSolverIterations=self.numSolverIterations,
                                      numSubSteps=self.frame_skip)

  def step(self, frame_skip):
    self._p.stepSimulation()    # 引擎迭代
