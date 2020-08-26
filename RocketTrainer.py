# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 13:53:10 2020

@author: Tobias
"""
from typing import List
import tensorflow as tf
import pickle

import matplotlib.pyplot as plt
import numpy as np
#from stable_baselines import SAC
from stable_baselines import TD3
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.evaluation import evaluate_policy
#from stable_baselines.ddpg.noise import NormalActionNoise
from stable_baselines.common.noise import NormalActionNoise
from stable_baselines.td3.policies import MlpPolicy, LnMlpPolicy
from stable_baselines.td3.policies import LnCnnPolicy, CnnPolicy
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.callbacks import CheckpointCallback, EveryNTimesteps

from LearningRocket import LearningRocket
from NormalizedActions import NormalizeActionWrapper
#from stable_baselines.gail import generate_expert_traj
#from stable_baselines.gail import ExpertDataset
from DummyExpert import DummyExpert

class RocketTrainer:

    def __init__(self, algorithm="SAC", load=True, agent_name="Agent001"):
        self.agent_name = agent_name

        self.env = LearningRocket(visualize=False)
        self.env = NormalizeActionWrapper(self.env)

        self.eval_env = LearningRocket(visualize=True)
        self.eval_env = NormalizeActionWrapper(self.eval_env)
        self.eval_callback = EvalCallback(self.eval_env, best_model_save_path='Agent007',
                                     log_path='./logs/', eval_freq=100000,
                                     deterministic=True, render=False,n_eval_episodes=1)
        kai_policy = dict(act_fun=tf.nn.tanh, net_arch=[400, 300])
        #check_env(self.env, warn=True)
        """
        if algorithm == "SAC":
            if load is True:
                self.model = SAC.load(agent_name, env=self.env, tensorboard_log="./rocket_tensorboard/")
                #self.model.ent_coef=0.2
            else:
                self.model = SAC('MlpPolicy', self.env, verbose=1, tensorboard_log="./rocket_tensorboard/",ent_coef=5)
            print("Trainer Set for SAC")
        """
        if algorithm == "TD3":
            n_actions = self.env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
            if load is True:
                self.model = TD3.load(agent_name, env=self.env, tensorboard_log="./rocket_tensorboard/")
                #file = open('replay_buffer', 'rb')
                #self.model.replay_buffer = pickle.load(file)
                #file.close()
            else:
                self.model = TD3(MlpPolicy, self.env, action_noise=action_noise, batch_size=128, gamma = 0.99,
                                learning_rate=2.5e-4, learning_starts=20000, verbose=1, tensorboard_log="./rocket_tensorboard/", policy_kwargs = dict(layers=[400, 300]))
            print("Trainer Set for TD3")

    def train(self, visualize=False, lesson_length=100000,lessons=1):
        print("Today I'm teaching rocket science. How hard can it be?")
        #self.env.render(visualize)
        for i in range(lessons):
            print("*sigh* here we go again.")
            self.model.learn(total_timesteps=lesson_length,callback=self.eval_callback)#,callback=self.eval_callback)
            self.model.save(self.agent_name)
            a_file = open('replay_buffer', 'wb')
            pickle.dump(self.model.replay_buffer, a_file)
            a_file.close()
            print("{} Batches Done.".format(i+1))
            # plt.close()
            mean_reward, std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=10)
            print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
        self.evaluate()

    def lecture(self):
        teacher = DummyExpert()
        #teacher = NormalizeActionWrapper(teacher)
        print("Let me show you how it's done.")
        generate_expert_traj(teacher.teach, 'dummy_expert_rocket', self.env, n_episodes=10)

    def evaluate(self):
        print("Watch this!")
        obs = self.eval_env.reset()
        #self.eval_env.render(True)

        reward_list = []
        reward_sum: List[float] = []
        action_list = []
        for i in range(3):
            action_list.append([])
        Time = []
        steps = 0
        cumulativeReward = 0
        data=[]
        for i in range(obs.size):
            data.append([])

        done = False
        while done is False:
            action, states = self.model.predict(obs)
            obs, reward, done, info = self.eval_env.step(action)
            re_obs = self.eval_env.rescale_observation((obs))
            #obs = self.eval_env.rescale_observation(obs)
            action = self.eval_env.rescale_action(action)
            reward_list.append(reward)
            cumulativeReward += reward
            reward_sum.append(cumulativeReward)
            action_list[0].append(action[0])
            #for i in range(3):
            #    action_list[i].append(action[i])
            for i in range(obs.size):
                data[i].append(re_obs[i])
            steps += 1
            Time.append(steps)

        print("Another happy landing.")

        plt.figure(figsize=(11, 8))
        plt.subplot(3, 2, 3)
        plt.xlabel('Time(s)')
        plt.ylabel('Position (m)')
        plt.plot(Time, data[0], label='X Position')
        plt.plot(Time, data[1], label='Speed')
        #plt.plot(Time, data[2], label='Z Position')
        plt.legend(loc='best')
        plt.subplot(3, 2, 1)
        plt.xlabel('Time(s)')
        plt.ylabel('Reward')
        plt.plot(Time, reward_list, label='Reward')
        plt.plot(Time, reward_sum, label='Total Reward')
        plt.legend(loc='best')
        plt.subplot(3, 2, 2)
        plt.xlabel('Time(s)')
        plt.ylabel('Actions')
        plt.plot(Time, action_list[0], label='Thrust')
        #plt.plot(Time, action_list[1], label='GimbalX')
        #plt.plot(Time, action_list[2], label='GimbalY')
        plt.legend(loc='best')

        plt.subplot(3, 2, 4)
        plt.xlabel('Time(s)')
        plt.ylabel('Attitude')
        #plt.plot(Time, data[4], label='Roll')
        #plt.plot(Time, data[4], label='Pitch')
        #plt.plot(Time, data[5], label='Yaw')
        plt.legend(loc='best')

        plt.subplot(3, 2, 5)
        plt.xlabel('Time(s)')
        plt.ylabel('Velocity')
        #plt.plot(Time, data[2], label='vX')
        #plt.plot(Time, data[3], label='vY')
        #plt.plot(Time, data[5], label='vZ')
        plt.legend(loc='best')

        plt.subplot(3, 2, 6)
        plt.xlabel('Time(s)')
        plt.ylabel('RotVel')
        #plt.plot(Time, data[12], label='Fuel')
        #plt.plot(Time, data[6], label='Rot X')
        #plt.plot(Time, data[7], label='Rot Y')
        plt.legend(loc='best')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    T = RocketTrainer(algorithm="TD3", load=True, agent_name="Doof")
    T.train(visualize=False, lesson_length=500000, lessons=1)
    #T.env.render(True)
    #T.lecture()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              T.evaluate()
    #data_set = ExpertDataset(expert_path='dummy_expert_rocket.npz',batch_size=128)
    #T.model.pretrain(data_set, n_epochs=10000)
    #T.model.save(T.agent_name)
    #T.evaluate()
