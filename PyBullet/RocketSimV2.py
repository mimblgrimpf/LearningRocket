# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:10:54 2020

@author: Tobias
"""

import pybullet as p
import time
import pybullet_data


physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.81)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,2]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("rocket.urdf",cubeStartPos, cubeStartOrientation)

for i in range (10000):
    loc = [0,0,-0.15]
    force = [0,0,250]
    torque = [0,i/240,0]
    
    #print(force)
    p.applyExternalForce(boxId,1,force,loc,p.LINK_FRAME)
    #p.applyExternalTorque(boxId,-1,torque,p.LINK_FRAME)
    
    p.setJointMotorControl(boxId,4,p.POSITION_CONTROL,0.5)
    if i > 240*1:
        p.setJointMotorControl(boxId,4,p.POSITION_CONTROL,-0.5)
    
    cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    """if cubePos[2] > 10:
        p.changeDynamics(boxId, -1, mass=50)
        p.changeDynamics(boxId, 0, )"""
    #p.resetDebugVisualizerCamera(5,30,-30,cubePos)
    p.stepSimulation()
    time.sleep(1./240.)
    
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()