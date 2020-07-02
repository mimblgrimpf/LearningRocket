# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:10:54 2020

@author: Tobias
"""

import time
import pybullet as p
import pybullet_data
import PID
import random as r
import numpy as np
from scipy.spatial.transform import Rotation as rot

import RocketEngine as RE



#Flags
syntheticCamera = 0
RCSEnable = 1
realTime = 0

#Paramters
timestep = 60
thrustAtNozzle = [0,0,-0.15]
landedHeight = 1.5



R = RE.RocketEngine
RCS = RE.RocketEngine(100*9.81,150*9.81,10,0)
rcs_right = 6
rcs_left =7
rcs_front = 8 
rcs_rear = 9


physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setTimeStep(1/timestep)
p.setRealTimeSimulation(realTime)
p.setGravity(0,0,-9.81)
planeId = p.loadURDF("plane.urdf")
rocketStartPos = [r.randrange(-20,20,1),0,20]
rocketStartOrientation = p.getQuaternionFromEuler([0,0,0])
rocket = p.loadURDF("rocketwithrcs.urdf",rocketStartPos, rocketStartOrientation)
 

p.setRealTimeSimulation(0)


p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,syntheticCamera)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,syntheticCamera)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,syntheticCamera)




heightPID = PID.PID(0.5,0,3,0,1)
pitchPID = PID.PID(1,0.02,2,-0.2,0.2)
XPID = PID.PID(0.01,0,0.02,-1,1)


#Debug
drymass = p.getDynamicsInfo(rocket,-1)[0] + p.getDynamicsInfo(rocket,1)[0] + p.getDynamicsInfo(rocket,2)[0]*4


throttleInput = p.addUserDebugParameter("Throttle Position",0.4,1,0.4)
nozzleInput = p.addUserDebugParameter("Nozzle Position",-0.2,0.2,0)

timestamp = time.time()



for i in range (100000):
    
    rocketPos, rocketOrn = p.getBasePositionAndOrientation(rocket)
    rocketYPR = p.getEulerFromQuaternion(rocketOrn)
    #print(rocketYPR)
    #print(rocketPos)
    rocketVel = p.getBaseVelocity(rocket)
    
    
    #Controller
    throttle = heightPID.control(rocketPos[2],rocketVel[0][2],1.5)
    pitchTgt = XPID.control(rocketPos[0],rocketVel[0][0],0)
    nozzleTgt = pitchPID.control(rocketYPR[1],rocketVel[1][1],pitchTgt)
    #print(nozzleTgt)
    
    
    #Debug
    
    if i < 1200:
        #Call Engine Model, apply thrust
        thrust,mdot = R.setThrottle(R,throttle)
    else:
        thrust,mdot = R.setThrottle(R,p.readUserDebugParameter(throttleInput))
        p.setJointMotorControl(rocket,1,p.POSITION_CONTROL,p.readUserDebugParameter(nozzleInput))
    
    
    thrust,mdot = R.setThrottle(R,throttle)
    
    
    p.setJointMotorControl(rocket,1,p.POSITION_CONTROL,(-nozzleTgt)*0.2)
    p.applyExternalForce(rocket,1,thrust,thrustAtNozzle,p.LINK_FRAME)
    
    
    
    
    
    
    #RCS
    if RCSEnable:
        RCSThrust,RCSmdot = RCS.setThrottle(nozzleTgt)
        mdot = mdot + RCSmdot
        if nozzleTgt < 0:
            p.applyExternalForce(rocket,rcs_front,RCSThrust,[0,0,0],p.LINK_FRAME)
            
            lineStuff = p.getLinkState(rocket, rcs_front)
            
        elif nozzleTgt > 0:
            p.applyExternalForce(rocket,rcs_rear,RCSThrust,[0,0,0],p.LINK_FRAME)
            
            lineStuff = p.getLinkState(rocket, rcs_rear)
            
        
        
        
        
    
    if (time.time()-timestamp) > 0.05:
        timestamp = time.time()
        
        thrustLineStuff = p.getLinkState(rocket, 1)
        thrustlineStart = list(thrustLineStuff)[0]
        thrustlineOrn = list(thrustLineStuff)[1]
        
        thrustlineEnd = np.dot(thrust,-0.01)
        thrustlineRot = rot.from_quat(thrustlineOrn)
        thrustlineEnd = thrustlineRot.apply(thrustlineEnd)
        
        
        lineStart = list(lineStuff)[0]
        lineOrn = list(lineStuff)[1]
        
        if(nozzleTgt) < 0:
            lineEnd = np.dot(RCSThrust,-1)
        else:
            lineEnd = RCSThrust
        lineRot = rot.from_quat(lineOrn)
        lineEnd = lineRot.apply(lineEnd)
        
        p.addUserDebugLine(thrustlineStart,thrustlineStart+thrustlineEnd,[1,0.5,0],5,1/20)
        p.addUserDebugLine(lineStart,lineStart+lineEnd,[1,0.5,0],2,1/20)
        
        
    #Update Fuel State
    fuelMass = p.getDynamicsInfo(rocket,0)
    fuelMass = fuelMass[0] - (mdot/timestep)
    p.changeDynamics(rocket, 0, mass=fuelMass)
    
    pos = -1+(fuelMass/10)
    p.setJointMotorControl(rocket,0,p.POSITION_CONTROL,pos)
    
    fuelLength = 2*fuelMass/10
    fuelInertia = [1/12*fuelMass*(3*0.2*0.2+fuelLength*fuelLength),
                   1/12*fuelMass*(3*0.2*0.2+fuelLength*fuelLength),
                   1/2*fuelMass*0.2*0.2]
    p.changeDynamics(rocket,0,localInertiaDiagonal=fuelInertia)
    
        
    
    #Simulate
    p.resetDebugVisualizerCamera(4,30,-30,rocketPos)
    p.stepSimulation()
    time.sleep(1./timestep)
    
rocketPos, rocketOrn = p.getBasePositionAndOrientation(rocket)
print(rocketPos,rocketOrn)
p.disconnect()