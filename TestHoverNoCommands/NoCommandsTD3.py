import gym
import os

from stable_baselines.td3 import MlpPolicy

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from stable_baselines.common.callbacks import EvalCallback

#from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env, set_global_seeds
from stable_baselines import PPO2
from stable_baselines import TD3
from TestHoverNoCommands.LearningRocketHover import LearningRocket
import matplotlib.pyplot as plt
from stable_baselines.common.vec_env import VecNormalize, SubprocVecEnv
#from stable_baselines import PPO1
import time as t
import numpy as np
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise



def make_env(env_create_fkt, env_config, rank, seed=0):

    def _init():
        env = env_create_fkt(env_config)
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)
    return _init



def RocketTrainer():
    #env = SubprocVecEnv([make_env(LearningRocket, 'E:\Tobi\LearningRocket\TestHoverTD3\LearningRocketHover.py', i) for i in range(72)])

    # multiprocess environment
    env = make_vec_env(LearningRocket,n_envs=1)
    #env = LearningRocket(visualize=False)
    eval_env = make_vec_env(lambda: LearningRocket(visualize=True),n_envs=1)
    #env = VecNormalize(env)
    #eval_env = VecNormalize(eval_env)

    #env = VecNormalize.load("TestHoverTD3_env",env)
    #eval_env = VecNormalize.load("TestHoverTD3_env",eval_env)

    eval_callback = EvalCallback(eval_env, best_model_save_path='Agent007',
                                         log_path='./logs/', eval_freq=10000,
                                         deterministic=True, render=False,n_eval_episodes=1)

    #model = PPO2(MlpPolicy, env, n_steps=1500, nminibatches=144, lam=0.98, gamma=0.999, learning_rate=2.5e-4,
    #                                  noptepochs=4,ent_coef=0.01,verbose=1, tensorboard_log="./rocket_tensorboard/",
    #                                  policy_kwargs = dict(layers=[400, 300]))

    #model = PPO1(MlpPolicy, env, lam=0.98, gamma=0.999,verbose=1, tensorboard_log="./rocket_tensorboard/",
    #                                  policy_kwargs = dict(layers=[400, 300]))

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    model = TD3(MlpPolicy, env, action_noise=action_noise, batch_size=256, gamma = 0.95, target_policy_noise=0.01,
                target_noise_clip=0.02, train_freq=10, gradient_steps=10, learning_rate=1e-3, learning_starts=7500,
                verbose=1, tensorboard_log="./rocket_tensorboard/", policy_kwargs = dict(layers=[400, 300]),
                buffer_size=100000)
    #model = TD3(MlpPolicy,env,verbose=1)

    start = t.time()

    #model = PPO2.load("TestHoverTD3", env=env, tensorboard_log="./rocket_tensorboard/")
    #model = TD3.load("TestHoverTD3", env=env, tensorboard_log="./rocket_tensorboard/")
    #while True:
    #model.learning_rate = 2.5e-3
    model.learn(total_timesteps=200000,callback=eval_callback)
    model.save("TestHoverTD3")
    #env.save("TestHoverTD3_env")
    del model # remove to demonstrate saving and loading

    duration = t.time()-start

    model = TD3.load("TestHoverTD3",env=eval_env)
    #model = PPO2.load("TestHoverTD3", env=eval_env)

    # Enjoy trained agent
    obs = eval_env.reset()
    data = []
    time = []
    actions = []
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
    lastValves = [0.15, 0.2, 0.15]

    for i in range(600):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = eval_env.step(action)
        # Or_obs = eval_env.get_original_obs()

        time.append(i)
        for j in range(10):
            data[j].append(obs[0][j] * modifiers[j])
        data[2][i] -= 100
        for j in range(3):
            actions[j].append(action[0][j])
        offset = abs(data[0][i] - data[1][i])
        # if offset < 10:
        #    alt_reward.append(1-offset/10)
        # else:
        alt_reward.append((offset / 2) / 1000)

        mixError = abs(data[6][i] - 5.5)
        mix_reward.append((mixError / 0.2) / 1000)
        if mixError > 0.3:
            mix_reward[i] -= 1

        tempError = abs(data[5][i] - 900)
        temp_reward.append((tempError / 30) / 1000)
        if tempError > 50:
            temp_reward[i] -= 1

        total_reward.append(alt_reward[i] + mix_reward[i] + temp_reward[i])

        if start is True:
            alt_cumu.append(alt_reward[i])
            mix_cumu.append(mix_reward[i])
            temp_cumu.append(temp_reward[i])
            total_cumu.append(total_reward[i])
            start = False
        else:
            alt_cumu.append(alt_reward[i] + alt_cumu[i - 1])
            mix_cumu.append(mix_reward[i] + mix_cumu[i - 1])
            temp_cumu.append(temp_reward[i] + temp_cumu[i - 1])
            total_cumu.append(total_reward[i] + total_cumu[i - 1])

    plt.figure(figsize=(11, 8))
    plt.subplot(4, 2, 1)
    plt.xlabel('Time(s)')
    plt.ylabel('Offset (m)')
    plt.plot(time, data[0], label='Z Position')
    plt.plot(time, data[1], label='Z Speed')

    plt.subplot(4, 2, 2)
    plt.xlabel('Time(s)')
    plt.ylabel('Actions')

    plt.plot(time, actions[0], 'b', label='LOX Command')
    plt.plot(time, actions[1], 'r', label='LH2 Command')
    plt.plot(time, actions[2], 'y', label='Mix Command')
    plt.legend(loc='best')

    plt.subplot(4, 2, 3)
    plt.xlabel('Time(s)')
    plt.ylabel('Engine State')
    plt.plot(time, data[5], label='Temp')
    plt.legend(loc='best')

    plt.subplot(4, 2, 5)
    plt.xlabel('Time(s)')
    plt.ylabel('Engine State')
    plt.plot(time, data[4], label='Pressure')
    plt.legend(loc='best')

    plt.subplot(4, 2, 4)
    plt.xlabel('Time(s)')
    plt.ylabel('Mixture')
    plt.plot(time, data[6], label='Mixture')
    plt.legend(loc='best')

    plt.subplot(4, 2, 6)
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

    plt.legend(loc='best')

    print(duration)
    plt.show()


if __name__ == "__main__":
    RocketTrainer()