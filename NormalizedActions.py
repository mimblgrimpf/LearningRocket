import gym
import numpy as np


class NormalizeActionWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Retrieve the action space
        action_space = env.action_space
        observation_space = env.observation_space
        assert isinstance(action_space,
                          gym.spaces.Box), "This wrapper only works with continuous action space (spaces.Box)"
        assert isinstance(observation_space,
                          gym.spaces.Box), "This wrapper only works with continuous action space (spaces.Box)"
        # Retrieve the max/min values
        self.action_low, self.action_high = action_space.low, action_space.high
        self.observation_low, self.observation_high = observation_space.low, observation_space.high


        # We modify the action space, so all actions will lie in [-1, 1]
        env.action_space = gym.spaces.Box(low=-1, high=1, shape=action_space.shape, dtype=np.float32)
        env.observation_space = gym.spaces.Box(low=-1, high=1, shape=observation_space.shape, dtype=np.float32)

        # Call the parent constructor, so we can access self.env later
        super(NormalizeActionWrapper, self).__init__(env)

    def rescale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)
        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        return self.action_low + (0.5 * (scaled_action + 1.0) * (self.action_high - self.action_low))

    def downscale_action(self, action):
        return ((action - self.action_low) / (self.action_high - self.action_low) * 2.0) - 1.0

    def downscale_observation(self, observation):
        return ((observation - self.observation_low) / (self.observation_high - self.observation_low) * 2.0) - 1.0

    def rescale_observation(self,observation):
        return self.observation_low + (0.5 * (observation + 1.0) * (self.observation_high - self.observation_low))

    def reset(self):
        """
        Reset the environment
        """
        # Reset the counter
        obs = self.env.reset()
        return self.downscale_observation(obs)

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        # Rescale action from [-1, 1] to original [low, high] interval
        rescaled_action = self.rescale_action(action)
        obs, reward, done, info = self.env.step(rescaled_action)
        return self.downscale_observation(obs), reward, done, info