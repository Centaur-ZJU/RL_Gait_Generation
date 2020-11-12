import pybullet as p
import time
import numpy as np
import pybullet_data
print(np.clip(1.37, 0, 1))
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadSDF("/home/chk/anaconda3/lib/python3.7/site-packages/pybullet_data/chk_plane_stadium.sdf")
cubeStartPos = [0,0,1.2]
cubeStartOrientation = p.getQuaternionFromEuler([-1.57,0,0])
boxId = p.loadMJCF("./data/bfurdf/ballfoot_centaur.xml",
                                    flags=p.URDF_USE_SELF_COLLISION |
                                        p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS |
                                        p.URDF_GOOGLEY_UNDEFINED_COLORS)
p.resetBasePositionAndOrientation(boxId[0], cubeStartPos, cubeStartOrientation)
for i in range (10000):
    p.stepSimulation()    
    time.sleep(1./240.)
p.disconnect()
