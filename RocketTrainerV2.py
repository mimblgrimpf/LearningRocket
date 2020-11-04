import gym
from stable_baselines.common.callbacks import EvalCallback

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from LearningRocket import LearningRocket
import matplotlib.pyplot as plt
from stable_baselines.common.vec_env import VecNormalize
import numpy as np


# multiprocess environment
env = make_vec_env(LearningRocket,n_envs=16)
eval_env = make_vec_env(lambda: LearningRocket(visualize=True),n_envs=1)
#env = VecNormalize(env)
#eval_env = VecNormalize(eval_env)

env = VecNormalize.load("doof_env",env)
eval_env = VecNormalize.load("doof_env",eval_env)

eval_callback = EvalCallback(eval_env, best_model_save_path='Agent007',
                                     log_path='./logs/', eval_freq=10000,
                                     deterministic=True, render=False,n_eval_episodes=1)

#model = PPO2(MlpPolicy, env, n_steps=1000, nminibatches=32, lam=0.98, gamma=0.999, learning_rate=1e-4,
#                                  noptepochs=4,ent_coef=0.01,verbose=1, tensorboard_log="./rocket_tensorboard/",
#                                  policy_kwargs = dict(layers=[400, 300]))

"""model = PPO2(MlpPolicy, env,verbose=1, tensorboard_log="./rocket_tensorboard/",
                                  policy_kwargs = dict(layers=[400, 300]))"""

#model = PPO2.load("doof", env=env, tensorboard_log="./rocket_tensorboard/")
#model.learning_rate = 2.5e-4
#model.n_steps = 100
#while True:
#    model.learn(total_timesteps=5000000,callback=eval_callback   )
#    model.save("doof")
#    env.save("doof_env")

#del model # remove to demonstrate saving and loading

model = PPO2.load("doof", env=eval_env)

# Enjoy trained agent
obs = eval_env.reset()
data=[]
time=[]
actions=[]
alt_reward = []
mix_reward = []
temp_reward = []
valveChange = []

for i in range(16):
    data.append([])
for i in range(4):
    actions.append([])
lastValves = [0.15,0.2,0.15]


for i in range(1000):
    action, _states = model.predict(obs,deterministic=True)
    obs, rewards, dones, info = eval_env.step(action)
    Or_obs = eval_env.get_original_obs()

    ValveChange = abs(Or_obs[0][8] - lastValves[0]) + abs(Or_obs[0][9] - lastValves[1]) + abs(
        Or_obs[0][10] - lastValves[2])
    valveChange.append(ValveChange * 50)
    lastValves = [Or_obs[0][8], Or_obs[0][9], Or_obs[0][10]]

    time.append(i)
    for j in range(16):
        data[j].append(Or_obs[0][j])
    for j in range(4):
        actions[j].append(action[0][j])
    alt_reward.append(abs(Or_obs[0][0])/10)
    mix_reward.append(abs(Or_obs[0][6]/Or_obs[0][5]-5.5)*2)
    temp_reward.append(abs(Or_obs[0][4]-900)/1000)

plt.figure(figsize=(11, 8))
plt.subplot(4, 2, 1)
plt.xlabel('Time(s)')
plt.ylabel('Offset (m)')
plt.plot(time, data[0], label='Z Position')
plt.plot(time, data[1], label='Z Speed')


plt.subplot(4,2,2)
plt.xlabel('Time(s)')
plt.ylabel('Actions')

plt.plot(time, data[8], ':b', label='LOX Valve')
plt.plot(time, data[9], ':r', label='LH2 Valve')
plt.plot(time, data[10], ':y', label='Mix Valve')

plt.plot(time, actions[0], 'b', label='LOX Command')
plt.plot(time, actions[1], 'r', label='LH2 Command')
plt.plot(time, actions[2], 'y', label='Mix Command')
plt.legend(loc='best')



plt.subplot(4,2,3)
plt.xlabel('Time(s)')
plt.ylabel('Engine State')
plt.plot(time, data[3], label='Pressure')
plt.plot(time, data[4], label='Temp')
plt.legend(loc='best')

plt.subplot(4,2,4)
plt.xlabel('Time(s)')
#plt.ylabel('Fuel Flow')
#plt.plot(time, data[6], label='LOX Flow')
#plt.plot(time, data[5], label='LH2 Flow')
plt.ylabel('Mixture')
plt.plot(time, np.divide(data[6],data[5]), label='Mixture')
plt.legend(loc='best')

plt.subplot(4,2,5)
plt.xlabel('Time(s)')
plt.ylabel('Thrust')
plt.plot(time, data[7], label='Thrust')
plt.legend(loc='best')

plt.subplot(4,2,6)
plt.xlabel('Time(s)')
plt.ylabel('Reward values')
plt.plot(time, alt_reward, label='Altitude Error')
plt.plot(time, mix_reward, label='Mixture Error')
plt.plot(time, temp_reward, label='Temperature Error')
plt.plot(time,valveChange, label='ValveError')
plt.legend(loc='best')

plt.subplot(4,2,7)
plt.xlabel('Time(s)')
plt.ylabel('X Gimbal')
plt.plot(time, actions[3], label='X Gimbal Action')
plt.plot(time, data[11], label='X Pos')
plt.plot(time, data[12], label='Pos Speed')
plt.legend(loc='best')

plt.subplot(4,2,8)
plt.xlabel('Time(s)')
plt.ylabel('X Angles')
plt.plot(time, data[13], label='Yaw')
plt.plot(time, data[14], label='RotVel')
plt.legend(loc='best')

plt.show()