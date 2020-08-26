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

from PandaRocket import Simulation


def mag(vec):
    return abs(vec.getX()) + abs(vec.getY()) + abs(vec.getZ())


class LearningRocket(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, visualize=False):
        super(LearningRocket, self).__init__()
        self.sim = Simulation(visualize)
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        #self.action_space = spaces.Box(low=np.array([0.1, -10, -10]), high=np.array([1, 10, 10]), dtype=np.double)
        self.action_space = spaces.Box(low=np.array([0.1]), high=np.array([1]), dtype=np.double)
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
            low=np.array([0, -1, 0]),
            high=np.array([1, 1, 1]),
            dtype=np.double)


        # base.run()

    def step(self, action):
        self.sim.control(action[0])#, action[1], action[2])
        # print("In LR: {}".format(action[0]))
        # self.sim.update()

        pos, vel, Roll, Pitch, Yaw, rotVel, fuel, EMPTY, done, LANDED = self.sim.observe()
        """observation = np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], Roll, Pitch, Yaw, rotVel[0], rotVel[1],
                                rotVel[2], fuel])"""

        #observation = np.array([pos[0], pos[1], vel[0], vel[1], Pitch, Yaw, rotVel[0], rotVel[1]])
        observation = np.array([pos[2]/1000,vel[2]/100,fuel])

        if LANDED is True:

            #reward = -100.0 * (mag(pos) + 10.0 * mag(vel) + abs(Pitch) + abs(Yaw) + mag(rotVel)) / 100000.0
            #reward = -(abs(pos.getX()) + abs(pos.getY())) / 400.0 /2.0
            #reward = -1000

            reward = -abs(pos.getZ() - 250) / 500
        else:
            # reward = -0.01 * (abs(Pitch) + abs(Yaw) + mag(rotVel) + abs(pos.getX()) + abs(pos.getY())) / 100000.0
            #reward = -(abs(pos.getX()) + abs(pos.getY())) / 400.0 /2.0
            #reward = -abs(pos.getZ()-45)/500+1-np.exp((30-pos.getZ())*0.5)*10
            reward = -abs(pos.getZ() - 250) / 500
            # reward = 0
        info = {
            "A": "B"
        }
        return observation, reward, done, info

    def reset(self):
        self.sim.doReset()
        pos, vel, Roll, Pitch, Yaw, rotVel, fuel, EMPTY, done, LANDED = self.sim.observe()
        #observation = np.array([pos[0], pos[1], pos[2], Roll, Pitch, Yaw])
        #observation = np.array([pos[0], pos[1], vel[0], vel[1], Pitch, Yaw, rotVel[0], rotVel[1]])
        observation = np.array([pos[2]/1000, vel[2]/100,fuel])

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
