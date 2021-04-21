from ur10_env import UR10

import gym
import time
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2, SAC, HER
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import make_vec_env


class DoneOnSuccessWrapper(gym.Wrapper):
    """
    Reset on success and offsets the reward.
    Useful for GoalEnv.
    """
    def __init__(self, env, reward_offset=1.0):
        super(DoneOnSuccessWrapper, self).__init__(env)
        self.reward_offset = reward_offset

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        done = done or info.get('is_success', False)
        reward += self.reward_offset
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.env.compute_reward(achieved_goal, desired_goal, info)
        return reward + self.reward_offset


if __name__ == '__main__':
    env = DoneOnSuccessWrapper(UR10(is_train=False, is_dense=False))
    env.reset()
    for i in range(1000000):
        state, reward, done, info = env.step(env.action_space.sample())
        time.sleep(0.1)

        if done:
            env.reset()

    env.close()
