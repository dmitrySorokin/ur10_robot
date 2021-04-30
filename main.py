from gym.wrappers import FlattenObservation

from ur10_env import UR10

import gym
import time
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2, SAC, HER
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import make_vec_env
import numpy as np
from tqdm import trange


class HardcodedPolicy(object):
    def __init__(self, position_bounds):
        self.step = 0
        self.position_bounds = position_bounds

    def act(self, state):
        state = state['observation']
        self.step += 1
        gripper_pos = state[:3]
        delta_pos = state[12:15]

        target_pos = np.zeros(4)
        if self.step < 100:
            delta = delta_pos
            target_pos[:3] = self.rescale(delta, self.position_bounds)
            target_pos[3] = -1
        else:
            delta = np.array([1, 0, 1]) - gripper_pos
            target_pos[:3] = self.rescale(delta, self.position_bounds)
            target_pos[3] = 1
        return target_pos

    def reset(self):
        self.step = 0

    def rescale(self, values, bounds):
        result = np.zeros_like(values)
        for i, (value, (lower_bound, upper_bound)) in enumerate(zip(values, bounds)):
            result[i] = value / (upper_bound - lower_bound)
        return result


class Agent(object):
    def __init__(self, model):
        self.model = model

    def act(self, state):
        pred = self.model.predict([state])[0][0]
        return pred

    def reset(self):
        pass


def evaluate(policy, env, nepisodes=100, viz=False):
    success = []
    reward = []
    for episode in trange(nepisodes):
        state = env.reset()
        policy.reset()
        reward.append(0)
        while True:
            state, rew, done, info = env.step(policy.act(state))
            reward[-1] += rew
            if viz:
                time.sleep(0.1)
            if done:
                success.append(info['is_success'])
                break
    env.close()
    print(reward)
    return np.mean(success), np.mean(reward)


def train_ppo(nsteps):
    train_env = SubprocVecEnv([lambda: FlattenObservation(UR10(is_train=True, is_dense=True))] * 8)
    model = PPO2(MlpPolicy, train_env,
                 verbose=1, tensorboard_log="log",
                 policy_kwargs={'layers': [256, 256, 256]},
                 )
    model.learn(total_timesteps=int(nsteps))
    model.save("ppo_model")


def train_sac(nsteps):
    train_env = FlattenObservation(UR10(is_train=True, is_dense=True))
    model = SAC('MlpPolicy', train_env,
                 verbose=1, tensorboard_log="log",
                 policy_kwargs={'layers': [256, 256, 256]},
                 )
    model.learn(total_timesteps=int(nsteps))
    model.save("sac_model")


def train_hersac(nsteps):
    train_env = UR10(is_train=True, is_dense=False)
    model = HER('MlpPolicy', train_env, SAC, verbose=1, tensorboard_log="log",
                 policy_kwargs={'layers': [256, 256, 256]},)

    model.learn(total_timesteps=int(nsteps))
    model.save("her_model")


if __name__ == '__main__':
    #train_sac(1e6)
    #train_ppo(1e6)
    #env = FlattenObservation(UR10(is_train=False, is_dense=False))
    #print('success rate', evaluate(Agent(PPO2.load('ppo_model')), env, viz=True))
    env = UR10(is_train=True, is_dense=True)
    print('success rate', evaluate(HardcodedPolicy(env.position_bounds), env, viz=False))
