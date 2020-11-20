import gym
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from stable_baselines.common.callbacks import EvalCallback

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env, set_global_seeds
from stable_baselines import PPO2
from TestHoverNoCommands.LearningRocketHover import LearningRocket
import matplotlib.pyplot as plt
from stable_baselines.common.vec_env import VecNormalize, SubprocVecEnv
#from stable_baselines import PPO1
import time as t
import numpy as np



def make_env(env_create_fkt, env_config, rank, seed=0):

    def _init():
        env = env_create_fkt(env_config)
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)
    return _init



def RocketTrainer():
    env = SubprocVecEnv([make_env(LearningRocket, 'E:\Tobi\LearningRocket\TestHover\LearningRocketHover.py', i) for i in range(72)])

    eval_env = make_vec_env(lambda: LearningRocket(visualize=True),n_envs=1)

    eval_callback = EvalCallback(eval_env, best_model_save_path='Agent007',
                                         log_path='./logs/', eval_freq=10000,
                                         deterministic=True, render=False,n_eval_episodes=1)

    model = PPO2(MlpPolicy, env, n_steps=1500, nminibatches=144, lam=0.98, gamma=0.999, learning_rate=5e-4, cliprange=0.3,
                                      noptepochs=4,ent_coef=0.01,verbose=1, tensorboard_log="./rocket_tensorboard/",
                                      policy_kwargs = dict(layers=[400, 300]))

    start = t.time()

    #model = PPO2.load("TestHover", env=env, tensorboard_log="./rocket_tensorboard/")
    model.learn(total_timesteps=10000000,callback=eval_callback)
    model.save("TestHover")
    del model # remove to demonstrate saving and loading

    duration = t.time()-start


    model = PPO2.load("TestHover", env=eval_env)

    # Enjoy trained agent
    obs = eval_env.reset()
    data=[]
    time=[]
    actions=[]
    alt_reward = []
    mix_reward = []
    temp_reward = []
    valveChange = []
    speedPunishes = []
    total_reward = []
    alt_cumu = []
    mix_cumu = []
    temp_cumu = []
    total_cumu = []
    start = True
    modifiers = [1000, 1000, 200, 1, 200, 2000, 10, 1000, 1500, 1]


    for i in range(10):
        data.append([])
    for i in range(3):
        actions.append([])
    lastValves = [0.15,0.2,0.15]


    for i in range(600):
        action, _states = model.predict(obs,deterministic=True)
        obs, rewards, dones, info = eval_env.step(action)
        #Or_obs = eval_env.get_original_obs()

        time.append(i)
        for j in range(10):
            data[j].append(obs[0][j]*modifiers[j])
        data[2][i] -= 100
        for j in range(3):
            actions[j].append(action[0][j])
        offset = abs(data[0][i]-data[1][i])
        #if offset < 10:
        #    alt_reward.append(1-offset/10)
        #else:
        alt_reward.append((offset/2)/1000)

        mixError = abs(data[6][i] - 5.5)
        mix_reward.append((mixError / 0.2) / 1000)
        #if mixError > 0.3:
        #    mix_reward[i] += 1

        tempError = abs(data[5][i]-900)
        temp_reward.append((tempError/30)/1000)
        #if tempError > 50:
        #    temp_reward[i] += 1

        total_reward.append(alt_reward[i]+mix_reward[i]+temp_reward[i])

        if start is True:
            alt_cumu.append(alt_reward[i])
            mix_cumu.append(mix_reward[i])
            temp_cumu.append(temp_reward[i])
            total_cumu.append(total_reward[i])
            start = False
        else:
            alt_cumu.append(alt_reward[i]+alt_cumu[i-1])
            mix_cumu.append(mix_reward[i]+mix_cumu[i-1])
            temp_cumu.append(temp_reward[i]+temp_cumu[i-1])
            total_cumu.append(total_reward[i]+total_cumu[i-1])

    plt.figure(figsize=(11, 8))
    plt.subplot(4, 2, 1)
    plt.xlabel('Time(s)')
    plt.ylabel('Offset (m)')
    plt.plot(time, data[0], label='Z Position')
    plt.plot(time, data[1], label='Z Speed')


    plt.subplot(4,2,2)
    plt.xlabel('Time(s)')
    plt.ylabel('Actions')

    plt.plot(time, actions[0], 'b', label='LOX Command')
    plt.plot(time, actions[1], 'r', label='LH2 Command')
    plt.plot(time, actions[2], 'y', label='Mix Command')
    plt.legend(loc='best')



    plt.subplot(4,2,3)
    plt.xlabel('Time(s)')
    plt.ylabel('Engine State')
    plt.plot(time, data[5], label='Temp')
    plt.legend(loc='best')

    plt.subplot(4, 2, 5)
    plt.xlabel('Time(s)')
    plt.ylabel('Engine State')
    plt.plot(time, data[4], label='Pressure')
    plt.legend(loc='best')

    plt.subplot(4,2,4)
    plt.xlabel('Time(s)')
    plt.ylabel('Mixture')
    plt.plot(time, data[6], label='Mixture')
    plt.legend(loc='best')

    plt.subplot(4,2,6)
    plt.xlabel('Time(s)')
    plt.ylabel('Reward values. Valve Error REAL valves')
    plt.plot(time, alt_reward, label='Altitude Error')
    plt.plot(time, mix_reward, label='Mixture Error')
    plt.plot(time, temp_reward, label='Temperature Error')
    plt.plot(time, total_reward, label='Total Reward')

    plt.subplot(4, 2, 8)
    plt.xlabel('Time(s)')
    plt.ylabel('Reward values cumulative')
    plt.plot(time, alt_cumu, label='Altitude Error')
    plt.plot(time, mix_cumu, label='Mixture Error')
    plt.plot(time, temp_cumu, label='Temperature Error')
    plt.plot(time, total_cumu, label='Total Reward')

    plt.subplot(4, 2, 7)
    plt.xlabel('Time(s)')
    plt.ylabel('Thrust in kN')
    plt.plot(time, data[7])

    plt.legend(loc='best')

    print(duration)
    plt.show()


if __name__ == "__main__":
    RocketTrainer()