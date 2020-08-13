# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 13:53:10 2020

@author: Tobias
"""

import numpy as np
from stable_baselines import TD3
from stable_baselines import SAC
from stable_baselines.common.policies import MlpPolicy
import matplotlib.pyplot as plt
from stable_baselines.common.env_checker import check_env

from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.td3.policies import MlpPolicy

from NormalizedActions import NormalizeActionWrapper

from LearningRocket import LearningRocket

env = LearningRocket(VISUALIZE=False)
env = NormalizeActionWrapper(env)
check_env(env, warn=True)


n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3(MlpPolicy, env, action_noise=action_noise, verbose=1)
#model = SAC('MlpPolicy', env, verbose=1)
model.load("sac_rocket")

obs = env.reset()
env.sim.VISUALIZE = True
done = False
actionList = []
obsList = []
rewardList = []
rewardSum = []
X = []
Y = []
Z = []
Time = []
steps = 0
cumulativeReward = 0
while done is False:
    action, states = model.predict(obs)
    #print(action)
    obs, reward, done, info = env.step(action)
    rewardList.append(reward)
    cumulativeReward+=reward
    rewardSum.append(cumulativeReward)
    X.append(obs[0])
    Y.append(obs[1])
    Z.append(obs[2])
    steps+=1
    Time.append(steps)
"""
for i in range(100):
    action, states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    rewardList.append(reward)
    cumulativeReward += reward
    rewardSum.append(cumulativeReward)
    X.append(obs[0])
    Y.append(obs[1])
    Z.append(obs[2])
    steps+=1
    Time.append(steps)
"""
plt.figure(figsize=(11,8))
plt.subplot(2, 1, 1)
plt.xlabel('Time(s)')
plt.ylabel('Position (m)')
plt.plot(Time, X, label='X Position')
plt.plot(Time, Y, label='Y Position')
plt.plot(Time, Z, label='Z Position')
plt.legend(loc='best')
plt.subplot(2, 1, 2)
plt.xlabel('Time(s)')
plt.ylabel('Reward')
plt.plot(Time, rewardList, label='Reward')
plt.plot(Time, rewardSum, label='Total Reward')
plt.legend(loc='best')

plt.tight_layout()
plt.show()
print("A for effort...")