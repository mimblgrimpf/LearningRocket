import numpy as np
from stable_baselines.common.noise import NormalActionNoise
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2, TD3
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.td3 import MlpPolicy

from LearningRocket import LearningRocket
from NormalizedActions import NormalizeActionWrapper

import optuna

n_cpu = 1

def optimize_TD3(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'buffer_size': trial.suggest_loguniform('buffer_size',1000,1000000),
        'learning_starts':trial.suggest_loguniform('learning_starts',1000,10000),
        'batch_size':trial.suggest_loguniform('batch_size',100,500),
        'train_freq':trial.suggest_loguniform('train_freq',1000,10000),
        'gradient_steps':trial.suggest_loguniform('gradient_steps',100,10000),

    }


def optimize_agent(trial):
    """ Train the model and optimise
        Optuna maximises the negative log likelihood, so we
        need to negate the reward here
    """
    model_params = optimize_TD3(trial)
    env = SubprocVecEnv([lambda: NormalizeActionWrapper(LearningRocket(visualize=False)) for i in range(n_cpu)])

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3(MlpPolicy, env, action_noise=action_noise,policy_kwargs = dict(layers=[400, 300]))
    model.learn(10000)

    rewards = []
    n_episodes, reward_sum = 0, 0.0

    obs = env.reset()
    step=0
    while n_episodes < 4:
        step+=1
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        reward_sum += reward
        if done:
            rewards.append(reward_sum)
            reward_sum = 0.0
            n_episodes += 1
            obs = env.reset()

    last_reward = np.mean(rewards)
    trial.report(-1 * last_reward,step)

    return -1 * last_reward


if __name__ == '__main__':
    study = optuna.create_study(study_name='RocketStudy', storage='sqlite:///params.db', load_if_exists=True)
    #study.optimize(optimize_agent, n_trials=1000, n_jobs=1)
    print(study.best_params)