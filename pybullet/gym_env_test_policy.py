from spinup.utils.test_policy import load_policy_and_env, run_policy
import pybullet as p
import time

physicsClient = p.connect(p.GUI)
fpath = '/home/chk/文档/Centaur/data/2020-05-11_ppo_HumanoidFlagrunBulletEnv/'
env, get_action = load_policy_and_env(fpath)
run_policy(env, get_action, render=True)