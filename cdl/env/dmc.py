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
        return self._get_obs()

    def step(self, action):
        self.time_step = self.env.step(action)
        obs = self._get_obs()
        reward = self.time_step.reward
        done = self.time_step.last()
        info = {}
        return obs, reward, done, info

    def _get_obs(self):
        obs = self.time_step.observation
        return np.concatenate([np.array(obs[key]) for key in obs])

    def render(self, mode='human'):
        return self.env.physics.render()

    def close(self):
        pass