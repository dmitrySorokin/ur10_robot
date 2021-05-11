import pybullet
import pybullet_data
import gym
import numpy as np


class UR10(gym.Env):
    observation_space = gym.spaces.Dict(dict(
        desired_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
        achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
        observation=gym.spaces.Box(-np.inf, np.inf, shape=(25,), dtype='float32'),
    ))

    action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=float)
    position_bounds = [(0.5, 1.0), (-0.25, 0.25), (0.7, 1)]

    def __init__(self, is_train, is_dense=False):
        self.connect(is_train)

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.is_dense = is_dense
        self.distance_threshold = 0.15

        self.planeId = None
        self.robot = None
        self.joints = []
        self.links = {}

        self.step_id = None
        self.object = None

        self.initial_joint_values = np.array([1.0, 0.0, 1.0])
        self.gripper_orientation = pybullet.getQuaternionFromEuler([np.pi, 0, 0])
        self.joint_values = None
        self.gripper_value = None

        self.target_position = np.array([1, 0, 1])

    def step(self, action):
        self.step_id += 1
        self.joint_values += np.array(action[:3]) * 0.1
        self.joint_values = np.clip(self.joint_values, -1, 1)
        self.gripper_value = 1 if action[3] > 0 else -1

        # end effector points down, not up (in case useOrientation==1)
        target_pos = self._rescale(self.joint_values, self.position_bounds)
        self.move_hand(target_pos, self.gripper_orientation, self.gripper_value)

        pybullet.stepSimulation()

        object_pos, object_orient = pybullet.getBasePositionAndOrientation(self.object)
        info = self.compute_info(action)
        return self.compute_state(), self.compute_reward(object_pos, self.target_position, info), self.is_done(), info

    def reset(self):
        self._reset_world()

        x_position = np.random.uniform(0.75, 1)
        y_position = np.random.uniform(-0.25, 0.25)

        # FIXME add rotation
        orientation = pybullet.getQuaternionFromEuler([0, 0, np.pi / 2])

        #self.object = pybullet.loadURDF('cube_small.urdf', [x_position, y_position, 0.6], orientation, globalScaling=0.75)

        self.object = pybullet.loadURDF('random_urdfs/000/000.urdf', [x_position, y_position, 0.6], orientation, globalScaling=1)
        pybullet.stepSimulation()

        return self.compute_state()

    def start_log_video(self, filename):
        pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, filename)

    def stop_log_video(self):
        pybullet.stopStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4)

    def render(self, mode='human'):
        pass

    def __del__(self):
        pybullet.disconnect()

    def compute_state(self):
        state = np.zeros(3 * 4 + 3 * 4 + 1)
        gripper_position, gripper_orientation, _, _, _, _, gripper_velocity, gripper_angular_velocity = \
            pybullet.getLinkState(self.robot, linkIndex=self.links['gripper_finger_joint'], computeLinkVelocity=True)
        state[:3] = gripper_position
        state[3:6] = pybullet.getEulerFromQuaternion(gripper_orientation)
        state[6:9] = gripper_velocity
        state[9:12] = gripper_angular_velocity

        object_pos, object_orient = pybullet.getBasePositionAndOrientation(self.object)
        object_velocity, object_angular_velocity = pybullet.getBaseVelocity(self.object)
        state[12:15] = np.asarray(object_pos) - gripper_position
        state[15:18] = pybullet.getEulerFromQuaternion(object_orient)
        state[18:21] = object_velocity
        state[21:24] = object_angular_velocity

        state[24] = self.compute_gripper_position()

        return {'observation': state, 'desired_goal': self.target_position, 'achieved_goal': object_pos}

    def compute_reward(self, achieved_goal, desired_goal, info):
        distance = np.linalg.norm(achieved_goal - desired_goal)
        if self.is_dense:
            gripper_position, gripper_orientation, _, _, _, _, gripper_velocity, gripper_angular_velocity = \
                pybullet.getLinkState(self.robot, linkIndex=self.links['gripper_finger_joint'],
                                      computeLinkVelocity=True)

            gripper_distance = np.linalg.norm(achieved_goal - np.asarray(gripper_position))
            return 1 - min(distance, 0.5) - gripper_distance
        else:
            return -(distance > self.distance_threshold).astype(np.float32)

    def is_done(self):
        object_pos, object_orient = pybullet.getBasePositionAndOrientation(self.object)
        return self.step_id == 200 or np.linalg.norm(object_pos - np.array([1.0, 0.0, 0.6])) > 1.0

    def compute_info(self, last_action):
        object_pos, object_orient = pybullet.getBasePositionAndOrientation(self.object)
        distance = np.linalg.norm(object_pos - self.target_position)
        return {
            'is_success': distance < self.distance_threshold,
            'gripper_pos': self.compute_gripper_position(),
            'last_action': last_action
        }

    def connect(self, is_train):
        if is_train:
            pybullet.connect(pybullet.DIRECT)
        else:
            pybullet.connect(pybullet.GUI)

    def _rescale(self, values, bounds):
        result = np.zeros_like(values)
        for i, (value, (lower_bound, upper_bound)) in enumerate(zip(values, bounds)):
            result[i] = (value + 1) / 2 * (upper_bound - lower_bound) + lower_bound
        return result

    def move_hand(self, target_position, orientation, gripper_value):
        joint_poses = pybullet.calculateInverseKinematics(
            self.robot,
            10, # 'gripper_finger_joint'
            target_position,
            orientation,
            maxNumIterations=100,
            residualThreshold=.01
        )

        for joint_id in range(6):
            pybullet.setJointMotorControl2(
                self.robot, self.joints[joint_id]['jointID'],
                pybullet.POSITION_CONTROL,
                targetPosition=joint_poses[joint_id],
            )

        for joint_id in range(6, 12):
            value = gripper_value
            minval, maxval = self.joints[joint_id]['jointLowerLimit'], self.joints[joint_id]['jointUpperLimit']
            value = (value + 1) / 2 * (maxval - minval) + minval
            pybullet.setJointMotorControl2(
                self.robot,
                self.joints[joint_id]['jointID'],
                pybullet.POSITION_CONTROL,
                targetPosition=value,
            )

    def compute_gripper_position(self):
        values = np.zeros(6)
        for i, joint_id in enumerate(range(6, 12)):
            data = pybullet.getJointState(self.robot, self.joints[joint_id]['jointID'])
            position, velocity = data[:2]
            lower_bound, upper_bound = self.joints[joint_id]['jointLowerLimit'], self.joints[joint_id]['jointUpperLimit']
            values[i] = (position - lower_bound) / (upper_bound - lower_bound) * 2 - 1
        return np.mean(values)

    def _reset_world(self):
        pybullet.resetSimulation()

        pybullet.setGravity(0, 0, -10)
        self.planeId = pybullet.loadURDF('plane.urdf')
        robot_position = [0, 0, 1.0]
        robot_orientation = pybullet.getQuaternionFromEuler([0, 0, 0])
        self.robot = pybullet.loadURDF('ur10_robot.urdf', robot_position, robot_orientation)

        joint_type = ['REVOLUTE', 'PRISMATIC', 'SPHERICAL', 'PLANAR', 'FIXED']
        self.joints = []
        self.links = {}

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
                self.links[data['jointName']] = joint_id

        self.step_id = 0
        self.object = None

        self.initial_joint_values = np.array([1.0, 0.0, 1.0])
        self.gripper_orientation = pybullet.getQuaternionFromEuler([np.pi, 0, 0])
        self.joint_values = self.initial_joint_values
        self.gripper_value = -1

        pybullet.loadURDF('table/table.urdf', globalScaling=1, basePosition=[0.5, 0, 0])

        self.position_bounds = [(0.5, 1.0), (-0.25, 0.25), (0.7, 1)]
        self.target_position = np.array([1, 0, 1])
