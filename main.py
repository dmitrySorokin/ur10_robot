from ur10_env import UR10

import pybullet
import pybullet_data
import gym
import time
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2, SAC
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import make_vec_env


def foo():
    env = UR10('train')
    train_env = DummyVecEnv([lambda : env]) # The algorithms require a vectorized environment to run

    model = PPO2(MlpPolicy, train_env, verbose=1, tensorboard_log="log", policy_kwargs={'layers': [256, 256]})
    model.learn(total_timesteps=int(1e7))
    model.save("ppo2_model")
    del model # remove to demonstrate saving and loading
    train_env.close()
    del train_env

    model = PPO2.load("ppo2_model")

    test_env = UR10('test')
    state = test_env.reset()
    for i in range(1000000):
        print(model.predict([state]))
        state, reward, done, info = test_env.step(model.predict([state])[0][0])
        time.sleep(2)

        if done:
            test_env.reset()

    test_env.close()


if __name__ == '__main__':
    env = UR10('test')
    #env.reset()
    for i in range(1000000):
        action = [0] * 12
        # TODO gripper action from -1, to 0; with center in -0.5
        # TODO initially gripper is open (in -1) -> 0, action = 1 closes gripper fully to 0.
        value = -0.3
        action[6:] = [value] * 6
        #state, reward, done, info = env.step(action)
        # time.sleep(2)

        #if done:
        #    env.reset()
        pybullet.stepSimulation()

    env.close()
