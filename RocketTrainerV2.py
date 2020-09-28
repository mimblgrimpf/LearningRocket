import gym
from stable_baselines.common.callbacks import EvalCallback

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from LearningRocket import LearningRocket
import matplotlib.pyplot as plt
from stable_baselines.common.vec_env import VecNormalize


# multiprocess environment
env = make_vec_env(LearningRocket,n_envs=16)
eval_env = make_vec_env(lambda: LearningRocket(visualize=True),n_envs=1)
#env = VecNormalize(env)
#eval_env = VecNormalize(eval_env)

env = VecNormalize.load("dick_env",env)
eval_env = VecNormalize.load("dick_env",eval_env)

eval_callback = EvalCallback(eval_env, best_model_save_path='Agent007',
                                     log_path='./logs/', eval_freq=10000,
                                     deterministic=True, render=False,n_eval_episodes=1)

"""model = PPO2(MlpPolicy, env, n_steps=1000, nminibatches=32, lam=0.98, gamma=0.999, learning_rate=1e-4,
                                  noptepochs=4,ent_coef=0.01,verbose=1, tensorboard_log="./rocket_tensorboard/",
                                  policy_kwargs = dict(layers=[400, 300]))"""

"""model = PPO2(MlpPolicy, env,verbose=1, tensorboard_log="./rocket_tensorboard/",
                                  policy_kwargs = dict(layers=[400, 300]))"""

model = PPO2.load("dick", env=env, tensorboard_log="./rocket_tensorboard/")
model.learn(total_timesteps=2000000,callback=eval_callback)
model.save("dick")
env.save("dick_env")

#del model # remove to demonstrate saving and loading

model = PPO2.load("dick")

# Enjoy trained agent
obs = eval_env.reset()
data=[]
time=[]
for i in range(1000):
    action, _states = model.predict(obs,deterministic=True)
    obs, rewards, dones, info = eval_env.step(action)
    time.append(i)
    data.append(obs[0][0])

plt.figure(figsize=(11, 8))
plt.subplot(3, 2, 3)
plt.xlabel('Time(s)')
plt.ylabel('Position (m)')
plt.plot(time, data, label='X Position')
plt.show()