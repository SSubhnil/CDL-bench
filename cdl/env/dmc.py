from dm_control import suite
import numpy as np

class DMCWrapper:
    def __init__(self, domain_name, task_name):
        self.env = suite.load(domain_name, task_name)
        self.action_spec = self.env.action_spec()
        self.observation_spec = self.env.observation_spec()

        # Define action_dim based on action_spec
        self.action_dim = self.action_spec.shape[0] if self.action_spec.shape else 1

    def reset(self):
        self.time_step = self.env.reset()
        raw_obs = self.time_step.observation
        return raw_obs, self._get_obs(raw_obs)

    def step(self, action):
        self.time_step = self.env.step(action)
        raw_obs = self.time_step.observation
        obs = self._get_obs(raw_obs)
        reward = self.time_step.reward
        done = self.time_step.last()
        info = {}
        return raw_obs, obs, reward, done, info

    def _get_obs(self, raw_obs):
        # Ensure this is the correct call for your environment
        if self.time_step is None:
            raise ValueError("Environment not reset. Call reset() before _get_obs().")
        # Debug: Print raw observations
        processed_obs = {}
        for key, value in raw_obs.items():
            if np.isscalar(value):
                processed_obs[key] = np.array([value])  # Convert scalar to single-element array
            else:
                processed_obs[key] = np.array(value).flatten()

        return processed_obs

    def render(self, mode='rgb_array'):
        return self.env.physics.render(mode=mode)

    def close(self):
        pass