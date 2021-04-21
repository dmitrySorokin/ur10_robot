import pybullet
import pybullet_data
import gym
import numpy as np


class UR10(gym.Env):
    gym.spaces.Dict()
    observation_space = gym.spaces.Dict(dict(
        desired_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
        achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
        observation=gym.spaces.Box(-np.inf, np.inf, shape=(29,), dtype='float32'),
    ))

    action_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=float)

    def __init__(self, is_train, is_dense=False):
        self.connect(is_train)

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
        pybullet.setGravity(0, 0, -10)
        self.planeId = pybullet.loadURDF('plane.urdf')
        robot_position = [0, 0, 1.0]
        robot_orientation = pybullet.getQuaternionFromEuler([0, 0, 0])
        self.robot = pybullet.loadURDF('ur10_robot_constr.urdf', robot_position, robot_orientation)

        joint_type = ['REVOLUTE', 'PRISMATIC', 'SPHERICAL', 'PLANAR', 'FIXED']
        self.joints = []
        for joint_id in range(pybullet.getNumJoints(self.robot)):
            info = pybullet.getJointInfo(self.robot, joint_id)
            data = {
                'jointID': info[0],
                'jointName': info[1].decode('utf-8'),
                'jointType': joint_type[info[2]],
                'jointLowerLimit': info[8],
                'jointUpperLimit': info[9],
                'jointMaxForce': info[10],
                'jointMaxVelocity': info[11]
            }
            if data['jointType'] != 'FIXED':
                self.joints.append(data)

        for i, joint in enumerate(self.joints):
            print(i, joint)

        self.is_dense = is_dense
        self.distance_threshold = 0.15

        self.step_id = None
        self.object = None
        self.joint_values = None
        self.tray_position = np.array([0.5, -1, 0])
        self.tray = pybullet.loadURDF('tray/traybox.urdf', self.tray_position)
        pybullet.loadURDF('table/table.urdf', globalScaling=1, basePosition=[0.5, 0, 0])

    def step(self, action):
        self.step_id += 1
        self.joint_values += np.array(action) * 0.1
        self.joint_values = np.clip(self.joint_values, -1, 1)

        for joint_id in range(5):
            value = self.joint_values[joint_id]
            minval, maxval = self.joints[joint_id]['jointLowerLimit'], self.joints[joint_id]['jointUpperLimit']
            value = (value + 1) / 2 * (maxval - minval) + minval
            pybullet.setJointMotorControl2(self.robot, self.joints[joint_id]['jointID'],
                                           pybullet.POSITION_CONTROL,
                                           targetPosition=value)

        gripper_value = self.compute_gripper_value()

        for joint_id in range(6, 12):
            value = gripper_value
            minval, maxval = self.joints[joint_id]['jointLowerLimit'], self.joints[joint_id]['jointUpperLimit']
            value = (value + 1) / 2 * (maxval - minval) + minval
            pybullet.setJointMotorControl2(self.robot, self.joints[joint_id]['jointID'],
                                           pybullet.POSITION_CONTROL,
                                           targetPosition=value)

        pybullet.stepSimulation()

        object_pos, object_orient = pybullet.getBasePositionAndOrientation(self.object)
        return self.compute_state(), self.compute_reward(object_pos, self.tray_position, {}), self.is_done(), self.compute_info()

    def reset(self):
        self.step_id = 0
        self.joint_values = np.zeros(self.action_space.shape)

        if self.object is not None:
            pybullet.removeBody(self.object)

        x_position = np.random.uniform(0.75, 1.2)
        y_position = np.random.uniform(-0.25, 0.25)
        orientation = pybullet.getQuaternionFromEuler([0, 0, np.random.uniform(0, 2 * np.pi)])

        self.object = pybullet.loadURDF('cube_small.urdf', [x_position, y_position, 0.6], orientation, globalScaling=1)

        return self.step([0] * self.action_space.shape[0])[0]

    def render(self, mode='human'):
        pass

    def close(self):
        pybullet.disconnect()

    def compute_state(self):
        state = np.zeros(12 * 2 + 4 + 1)
        for i, joint in enumerate(self.joints):
            data = pybullet.getJointState(self.robot, joint['jointID'])
            position, velocity = data[:2]
            state[i] = position
            state[i + 12] = velocity

        object_pos, object_orient = pybullet.getBasePositionAndOrientation(self.object)
        state[24: 28] = object_orient
        state[28] = self.compute_gripper_value()

        return {'observation': state, 'desired_goal': self.tray_position, 'achieved_goal': object_pos}

    def compute_reward(self, achieved_goal, desired_goal, info):
        distance =  np.linalg.norm(achieved_goal - desired_goal)
        if self.is_dense:
            return -distance
        else:
            return -(distance > self.distance_threshold).astype(np.float32)

    def is_done(self):
        return self.step_id == 200

    def compute_info(self):
        object_pos, object_orient = pybullet.getBasePositionAndOrientation(self.object)
        distance = np.linalg.norm(object_pos - self.tray_position)
        return {'is_success': distance < self.distance_threshold}

    def compute_gripper_value(self):
        if self.joint_values[5] <= 0:
           return -1
        return 0

    def connect(self, is_train):
        if is_train:
            pybullet.connect(pybullet.DIRECT)
        else:
             pybullet.connect(pybullet.GUI)
