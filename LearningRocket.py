# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 17:23:30 2020

@author: Tobias
"""
import math

import gym
import numpy as np
from direct.showbase.ShowBaseGlobal import globalClock
from gym import spaces

from PandaRocketLFOX import Simulation


def mag(vec):
    return abs(vec.getX()) + abs(vec.getY()) + abs(vec.getZ())


class LearningRocket(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    lastValves = [0.15,0.2,0.15]
    landingSpeedLimit = 0.5

    def __init__(self, visualize=False):
        super(LearningRocket, self).__init__()
        self.sim = Simulation(visualize)
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        #self.action_space = spaces.Box(low=np.array([0.1, -10, -10]), high=np.array([1, 10, 10]), dtype=np.double)
        self.action_space = spaces.Box(low=np.array([0.15,0.15,0.15,-10]), high=np.array([1,1,1,10]), dtype=np.double)
        # Example for using image as input:
        """self.observation_space = spaces.Box(
            low=np.array([-400, -400, 0, -100, -100, -100, -180, -180, -180, -2, -2, -2, 0]),
            high=np.array([400, 400, 1000, 100, 100, 100, 180, 180, 180, 2, 2, 2, 1]),
            dtype=np.double)"""
        """
        self.observation_space = spaces.Box(
            low=np.array([-400, -400, 0, -180, -180, -180]),
            high=np.array([400, 400, 1000, 180, 180, 180]),
            dtype=np.double)"""

        """
        self.observation_space = spaces.Box(
            low=np.array([-400,-400,-100,-100,-180,-180,-2,-2]),
            high=np.array([400,400,100,100,180,180,2,2]),
            dtype=np.double)"""

        self.observation_space = spaces.Box(
            low=np.array([-1000, -100, 0, -100, -100, -100, -100, -100,0.15,0.15,0.15,-500,100,-5,-5,0]),
            high=np.array([1000, 100, 1, 200, 2000, 100, 500, 2000,1,1,1,-500,100,5,5,1]),
            dtype=np.double)


        # base.run()

    def step(self, action):
        self.sim.control(np.array([action[0],action[1],action[2]]),action[3])#, action[1], action[2])
        # print("In LR: {}".format(action[0]))
        # self.sim.update()

        pos, vel, Roll, Pitch, Yaw, rotVel, fuel, EMPTY, done, LANDED, offset, EngObs, valves = self.sim.observe()
        """observation = np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], Roll, Pitch, Yaw, rotVel[0], rotVel[1],
                                rotVel[2], fuel])"""

        #observation = np.array([pos[0], pos[1], vel[0], vel[1], Pitch, Yaw, rotVel[0], rotVel[1]])
        #observation = np.array([pos[2],vel[2],fuel])
        observation = np.array([offset, vel[2], fuel,
                                EngObs[0], EngObs[1], EngObs[2], EngObs[3], EngObs[4],
                                valves[0],valves[1],valves[2],
                                pos.getX(),vel.getX(),Yaw,rotVel.getY(),
                                LANDED])

        #print(action[0],valves[0])
        #print(observation)

        #Height Control
        reward = -abs(offset) / 10

        #Position Control
        reward -= abs(pos.getX()) /10

        #Mixture Control
        mixture = EngObs[3]/EngObs[2]
        reward -= abs(mixture-5.5) *2

        #Temp Control
        reward -= abs(EngObs[1]-900) / 1000

        #Don't jitter the Valves
        ValveChange = abs(valves[0]-self.lastValves[0])+abs(valves[1]-self.lastValves[1])+abs(valves[2]-self.lastValves[2])
        #ValveChange = abs(action[0]-valves[0])+abs(action[1]-valves[1])+abs(action[2]-valves[2])
        reward -= ValveChange*5
        self.lastValves = valves

        #Don't blow up the engine
        if EngObs[1] > 900:
            reward -= 10

        """
        landingSpeed = abs(vel[0])+abs(vel[1])+abs(vel[2])
        landingDiff = landingSpeed-self.landingSpeedLimit

        if LANDED == 1:
            if landingDiff > 0:
                reward -= landingDiff*10000
            reward += 1
            valveDiff = valves[0]+valves[1]+valves[2]-0.5
            reward -= valveDiff
        """
        info = {
            "A": "B"
        }
        return observation, reward, done, info

    def reset(self):
        self.sim.doReset()
        pos, vel, Roll, Pitch, Yaw, rotVel, fuel, EMPTY, done, LANDED, offset, EngObs,valves = self.sim.observe()
        #observation = np.array([pos[0], pos[1], pos[2], Roll, Pitch, Yaw])
        #observation = np.array([pos[0], pos[1], vel[0], vel[1], Pitch, Yaw, rotVel[0], rotVel[1]])
        #observation = np.array([pos[2], vel[2],fuel])
        observation = np.array([offset, vel[2], fuel,
                                EngObs[0], EngObs[1], EngObs[2], EngObs[3], EngObs[4],
                                valves[0],valves[1],valves[2],
                                pos.getX(),vel.getX(),Yaw,rotVel.getY(),
                                LANDED])

        return observation

    def render(self, visualize=False):
        #self.sim.setVisualization(visualize)
        ...

    def close(self):
        ...


if __name__ == "__main__":
    env = LearningRocket()
    observation = env.reset()
    done = False
    while done is False:
        observation, reward, done, info = env.step([0.1, 0, 0])
        # print(observation)
    time = globalClock.getFrameTime()
    print(time)
    print(observation)

    env.sim.VISUALIZE = True
    observation = env.reset()
    for i in range(1000):
        observation = env.step([0.1, 0, 0])
        # print(observation)
    time = globalClock.getFrameTime()
    print(time)
    print(observation)
    # observation = env.reset()

    # If the environment don't follow the interface, an error will be thrown
    # check_env(env, warn=True)