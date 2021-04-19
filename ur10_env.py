import pybullet
import pybullet_data
import gym
import time
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2, SAC
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import make_vec_env


class UR10(gym.Env):
    observation_space = gym.spaces.Box(low=0, high=1, shape=(31,), dtype=float)
    action_space = gym.spaces.Box(low=-1, high=1, shape=(12,), dtype=float)

    def __init__(self, mode):
        self.connect(mode)

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
        pybullet.setGravity(0, 0, -10)
        self.planeId = pybullet.loadURDF("plane.urdf")
        robot_position = [0, 0, 1.0]
        robot_orientation = pybullet.getQuaternionFromEuler([0, 0, 0])
        self.robot = pybullet.loadURDF("ur10_robot.urdf", robot_position, robot_orientation)

        jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joints = []
        for joint_id in range(pybullet.getNumJoints(self.robot)):
            info = pybullet.getJointInfo(self.robot, joint_id)
            data = {
                'jointID': info[0],
                'jointName': info[1].decode('utf-8'),
                'jointType': jointTypeList[info[2]],
                'jointLowerLimit': info[8],
                'jointUpperLimit': info[9],
                'jointMaxForce': info[10],
                'jointMaxVelocity': info[11]
            }
            if data['jointType'] != 'FIXED':
                self.joints.append(data)

        for i, joint in enumerate(self.joints):
            print('\n', i, joint)
        # exit(0)

        self.step_id = None
        self.object = None
        self.joint_values = None
        self.tray = pybullet.loadURDF('tray/traybox.urdf', [0.5, -1, 0])
        pybullet.loadURDF('table/table.urdf', globalScaling=1, basePosition=[0.5, 0, 0])

    def step(self, action):
        self.step_id += 1
        #self.joint_values += np.asarray(action) * 0.05
        #self.joint_values = np.clip(self.joint_values, -1, 1)
        self.joint_values = np.asarray(action)
        for joint_id, value in enumerate(self.joint_values):
            minval, maxval = self.joints[joint_id]['jointLowerLimit'], self.joints[joint_id]['jointUpperLimit']
            value = (value + 1) / 2 * (maxval - minval) + minval
            pybullet.setJointMotorControl2(self.robot, self.joints[joint_id]['jointID'],
                                           pybullet.POSITION_CONTROL,
                                           targetPosition=value)
        pybullet.stepSimulation()

        return self.calc_state(), self.calc_reward(), self.is_done(), {}

    def reset(self):
        self.step_id = 0
        self.joint_values = np.zeros(12)
        if self.object is not None:
            pybullet.removeBody(self.object)
        x_position = np.random.uniform(0.75, 1.2)
        y_position = np.random.uniform(-0.25, 0.25)
        orientation = pybullet.getQuaternionFromEuler([0, 0, np.random.uniform(0, 2 * np.pi)])

        # self.object = pybullet.loadURDF(f'random_urdfs/{item}/{item}.urdf', [x_position, y_position, 0.75], globalScaling=1)
        self.object = pybullet.loadURDF('cube_small.urdf', [x_position, y_position, 0.6], orientation, globalScaling=1)

        for _ in range(100):
            pybullet.stepSimulation()

        return self.step([0] * 12)[0]
       #  return self.calc_state()

    def render(self, mode='human'):
        pass

    def close(self):
        pybullet.disconnect()

    def calc_state(self):
        state = np.zeros(12 * 2 + 3 + 4)
        for i, joint in enumerate(self.joints):
            data = pybullet.getJointState(self.robot, joint['jointID'])
            position, velocity = data[:2]
            state[i] = position
            state[i + 12] = velocity

        object_pos, object_orient = pybullet.getBasePositionAndOrientation(self.object)
        state[24: 27] = object_pos
        state[27:] = object_orient

        return state

    def calc_reward(self):
        object_pos, object_orient = pybullet.getBasePositionAndOrientation(self.object)
        return np.linalg.norm(object_pos - np.array([0.5, -1, 0])) - 2

        reward = -1
        if object_pos[2] > 0.75:
            reward += 10

        if np.linalg.norm([object_pos[:2] - np.array([0.5, -1])]) < 0.15:
            reward += 100
        elif object_pos[2] < 0.2:
            reward -= 100

        return reward

    def is_done(self):
        object_pos, object_orient = pybullet.getBasePositionAndOrientation(self.object)
        in_tray = np.linalg.norm([object_pos[:2] - np.array([0.5, -1])]) < 0.15
        return self.step_id == 100# or (object_pos[2] < 0.2 and not in_tray)

    def connect(self, mode):
        if mode == 'test':
            pybullet.connect(pybullet.GUI)
        elif mode == 'train':
             pybullet.connect(pybullet.DIRECT)
        else:
            raise ValueError(mode)


