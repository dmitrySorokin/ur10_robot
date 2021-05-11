from gym.wrappers import FlattenObservation
import gym
import time
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2, SAC, HER
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import make_vec_env
import numpy as np
from tqdm import trange
import argparse

from ur10_env import UR10


class HardcodedAgent(object):
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
            delta = delta_pos + np.array([0, 0, 1]) / self.step
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


class RLAgent(object):
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
                print(success[-1])
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, choices=['ppo', 'script'])
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'viz'])
    args = parser.parse_args()

    if args.mode == 'train':
        assert args.agent == 'ppo', 'script agent is not trainable'
        train_ppo(1e7)
    else:
        if args.agent == 'ppo':
            agent = RLAgent(PPO2.load('ppo_model.zip'))
            env = FlattenObservation(UR10(is_train=args.mode == 'eval', is_dense=True))
        else:
            agent = HardcodedAgent(UR10.position_bounds)
            env = UR10(is_train=args.mode == 'eval', is_dense=True)
        print('success rate', evaluate(agent, env, viz=args.mode != 'eval'))
