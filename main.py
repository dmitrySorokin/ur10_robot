from gym.wrappers import FlattenObservation
import time
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv
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
            target_pos[:3] = self._rescale(delta, self.position_bounds)
            target_pos[3] = -1
        else:
            delta = np.array([1, 0, 1]) - gripper_pos
            target_pos[:3] = self._rescale(delta, self.position_bounds)
            target_pos[3] = 1

        return target_pos

    def reset(self):
        self.step = 0

    def _rescale(self, values, bounds):
        result = np.zeros_like(values)
        for i, (value, (lower_bound, upper_bound)) in enumerate(zip(values, bounds)):
            result[i] = value / (upper_bound - lower_bound)
        return result


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state):
        return self.action_space.sample()

    def reset(self):
        pass


class RLAgent(object):
    def __init__(self, model):
        self.model = model

    def act(self, state):
        return self.model.predict([state])[0][0]

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


def make_video(policy, env, out='video.mp4'):
    state = env.reset()
    policy.reset()
    env.start_log_video(out)
    while True:
        state, rew, done, info = env.step(policy.act(state))
        time.sleep(0.1)
        if done:
            break
    env.stop_log_video()
    env.close()


def train_ppo(nsteps):
    train_env = SubprocVecEnv([lambda: FlattenObservation(UR10(is_train=True, is_dense=True))] * 8)
    model = PPO2(MlpPolicy, train_env,
                 verbose=1, tensorboard_log='log',
                 policy_kwargs={'layers': [256, 256, 256]},
                 )
    model.learn(total_timesteps=int(nsteps))
    model.save('ppo_model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, choices=['ppo', 'script', 'random'])
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'viz', 'video'])
    args = parser.parse_args()

    if args.mode == 'train':
        assert args.agent == 'ppo', f'{args.agent} agent is not trainable'
        train_ppo(2e5)
    else:
        if args.agent == 'ppo':
            env = FlattenObservation(UR10(is_train=args.mode == 'eval', is_dense=True))
            agent = RLAgent(PPO2.load('models/ppo_model.zip'))
        elif args.agent == 'script':
            env = UR10(is_train=args.mode == 'eval', is_dense=True)
            agent = HardcodedAgent(UR10.position_bounds)
        elif args.agent == 'random':
            env = UR10(is_train=args.mode == 'eval', is_dense=True)
            agent = RandomAgent(env.action_space)
        else:
            assert False, f'{args.agent} is not supported'

        if args.mode in ('eval', 'viz'):
            print('success rate', evaluate(agent, env, viz=args.mode != 'eval'))
        elif args.mode == 'video':
            make_video(agent, env)
        else:
            assert False, f'{args.mode} is not supported'
