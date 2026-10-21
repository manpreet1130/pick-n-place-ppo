import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time

class RobotEnvironment(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30
    }

    def __init__(self, render_mode=None):
        super().__init__()
        self.start_position = [0.002434193097806552, -0.7811217832664136, 0.009772560588829905, -2.062867921360752, 
                       -0.007282385107511599, 1.4879767515924243, 0.8049546850317025, 0.04, 0.04]
        self.active_joints = [0, 1, 2, 3, 4, 5, 6, 9, 10]
        self.hand_link = 11

        self.gripper_down = p.getQuaternionFromEuler([np.pi, 0, 0])

        self.max_episode_steps = 400
        self.current_step = 0
        self.max_action_step = 0.03
        self.distance_threshold = 0.03
        self.cube_half_dimensions = [0.03, 0.03, 0.03]
        self.target_half_dimensions = [0.1, 0.1, 0.01]

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # initialize the pybullet configurations
        if self.render_mode == "human":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # define environment
        self.table, self.robot, self.cube, self.target = self.loadScene()
        self.table_height = p.getAABB(self.table, -1)[1][2]
        self._lower_limits, self._upper_limits = self._getLimits()
        self.joint_ranges = [upper - lower for lower, upper in zip(self._lower_limits, self._upper_limits)]
        robot_position, _ = self._get_element_position_and_orientation(self.robot)
        # reachable area in front of the arm with the gripper pointing down (measured with IK)
        self.cube_min_limit_x, self.cube_max_limit_x = robot_position[0] + 0.33, robot_position[0] + 0.68
        self.cube_min_limit_y, self.cube_max_limit_y = robot_position[1] - 0.30, robot_position[1] + 0.30
        # initialize observation space and action space
        self.observation_dims = len(self.active_joints) + 13
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dims,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(4,),
            dtype=np.float32
        )

        self.render_width, self.render_height = 640, 480
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.0, 0.0, self.table_height + 0.2],
            distance=1.6,
            yaw=50,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )

        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.render_width / self.render_height,
            nearVal=0.1,
            farVal=10.0
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        p.setJointMotorControlArray(self.robot, 
                                    self.active_joints, 
                                    p.POSITION_CONTROL, 
                                    targetPositions=self.start_position)


        for i, joint in enumerate(self.active_joints):
            p.resetJointState(
                bodyUniqueId=self.robot,
                jointIndex=joint,
                targetValue=self.start_position[i],
                targetVelocity=0.0
            )

        random_x_position = self.np_random.uniform(self.cube_min_limit_x, self.cube_max_limit_x)
        random_y_position = self.np_random.uniform(self.cube_min_limit_y, self.cube_max_limit_y)
        target_position, _ = self._get_element_position_and_orientation(self.target)
        cube_no_zone_x = [target_position[0] - self.target_half_dimensions[0], target_position[0] + self.target_half_dimensions[0]]
        cube_no_zone_y = [target_position[1] - self.target_half_dimensions[1], target_position[1] + self.target_half_dimensions[1]]
        while cube_no_zone_x[0] <= random_x_position <= cube_no_zone_x[1] and cube_no_zone_y[0] <= random_y_position <= cube_no_zone_y[1]:
            random_x_position = self.np_random.uniform(self.cube_min_limit_x, self.cube_max_limit_x)
            random_y_position = self.np_random.uniform(self.cube_min_limit_y, self.cube_max_limit_y)

        p.resetBasePositionAndOrientation(self.cube, [random_x_position, random_y_position, self.table_height + self.cube_half_dimensions[2]], p.getQuaternionFromEuler([0, 0, 0]))
        p.resetBaseVelocity(self.cube, [0, 0, 0], [0, 0, 0])

        for _ in range(50):
            p.stepSimulation()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        self.current_step += 1
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        gripper_position, _ = self._get_gripper_position_and_orientation()
        current_target = np.array(gripper_position) + action[:3] * self.max_action_step

        current_target[2] = max(current_target[2], self.table_height + 0.01)

        # calculating and moving the revolute joints
        joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.robot,
            endEffectorLinkIndex=self.hand_link,
            targetPosition=current_target,
            targetOrientation=self.gripper_down,
            lowerLimits=self._lower_limits,
            upperLimits=self._upper_limits,
            jointRanges=self.joint_ranges,
            restPoses=self.start_position,
            maxNumIterations=100
        )

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.active_joints[:7],
            controlMode=p.POSITION_CONTROL,
            targetPositions=joint_positions[:7]
        )

        # gripper finger calibration
        if action[3] > 0:
            finger_target_position = 0.0
        else:
            finger_target_position = 0.04

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.active_joints[7:],
            controlMode=p.POSITION_CONTROL,
            targetPositions=[finger_target_position, finger_target_position],
            forces=[100, 100]
        )

        for _ in range(30):
            p.stepSimulation()
            if self.render_mode == 'human':
                time.sleep(1.0 / 240)

        observation = self._get_obs()
        info = self._get_info()
        reward = self.getReward(info)
        terminated = info["is_success"]
        truncated = self.current_step >= self.max_episode_steps

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            _, _, rgba, _, _ = p.getCameraImage(
                width=self.render_width,
                height=self.render_height,
                viewMatrix=self.view_matrix,
                projectionMatrix=self.projection_matrix,
                renderer=p.ER_TINY_RENDERER
            )

            rgba=np.asarray(rgba, dtype=np.uint8).reshape(self.render_height, self.render_width, 4)
            return rgba[:, :, :3]

        return None

    def close(self):
        if p.isConnected():
            p.disconnect()

    def getReward(self, info):
        if info["is_success"]:
            return 100.0

        gripper_position, _ = self._get_gripper_position_and_orientation()
        cube_position = info["cube_position"]

        goal_position = np.array([*info["target_position"][:2], self.table_height + self.cube_half_dimensions[2]])

        if info["grasped"]:
            # carry the cube to the target
            return -0.5 - np.linalg.norm(goal_position - cube_position)

        # hasn't reached cube yet, reach the cube
        return -2.0 - np.linalg.norm(cube_position - gripper_position)

    def isTerminated(self):
        return self._get_info()["is_success"]

    def _get_element_position_and_orientation(self, element):
        element_position, element_orientation = p.getBasePositionAndOrientation(element)
        return np.array(element_position, dtype=np.float32), np.array(element_orientation, dtype=np.float32)

    def _get_gripper_position_and_orientation(self):
        gripper_info = p.getLinkState(self.robot, self.hand_link)
        return np.array(gripper_info[0], dtype=np.float32), np.array(gripper_info[1], dtype=np.float32)

    def _get_obs(self):
        joint_positions = [state[0] for state in p.getJointStates(self.robot, self.active_joints)]
        gripper_position, _ = self._get_gripper_position_and_orientation()
        cube_position, _ = self._get_element_position_and_orientation(self.cube)
        target_position, _ = self._get_element_position_and_orientation(self.target)

        return np.concatenate([
            joint_positions,
            gripper_position,
            cube_position,
            cube_position - gripper_position,
            target_position - cube_position,
            [float(self._is_grasped())]
        ]).astype(np.float32)

    def _get_info(self):
        cube_position, _ = self._get_element_position_and_orientation(self.cube)
        target_position, _ = self._get_element_position_and_orientation(self.target)
        distance = np.linalg.norm(np.array(cube_position[:2]) - np.array(target_position[:2]))
        on_table = abs(cube_position[2] - (self.table_height + self.cube_half_dimensions[2])) < 0.005

        return {
            "cube_position": cube_position,
            "target_position": target_position,
            "distance_to_target": distance,
            "grasped": self._is_grasped(),
            "is_success": distance < self.distance_threshold and on_table
        }

    def loadScene(self):
        # plane
        p.loadURDF("plane.urdf")

        # table
        tablePosition, tableOrientation = [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0])
        table = p.loadURDF("table/table.urdf")

        table_min, table_max = p.getAABB(table, -1)

        # robot
        robotPosition, robotOrientation = [table_min[0] / 2.0, 0, table_max[2]], p.getQuaternionFromEuler([0, 0, 0])
        robot = p.loadURDF("franka_panda/panda.urdf", robotPosition, robotOrientation, useFixedBase=True)

        # target
        target_color = [1, 1, 0, 0.8]
        target_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=self.target_half_dimensions, rgbaColor=target_color)
        target_position, target_orientation = [table_max[0] / 4.0, 0, table_max[2]], p.getQuaternionFromEuler([0, 0, 0])
        target = p.createMultiBody(baseMass=0, 
                                   baseCollisionShapeIndex=-1, 
                                   baseVisualShapeIndex=target_visual,
                                   basePosition=target_position,
                                   baseOrientation=target_orientation)

        # cube
        cube_color = [1, 0, 0, 0.8]
        cube_collision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=self.cube_half_dimensions)
        cube_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=self.cube_half_dimensions, rgbaColor=cube_color)
        cube_position, cube_orientation = [table_max[0] / 4.0, table_max[1] / 4.0, table_max[2] + 0.03], p.getQuaternionFromEuler([0, 0, 0])
        cube = p.createMultiBody(baseMass=0.5, 
                                 baseCollisionShapeIndex=cube_collision,
                                 baseVisualShapeIndex=cube_visual,
                                 basePosition=cube_position,
                                 baseOrientation=cube_orientation)    
        return table, robot, cube, target

    def _getLimits(self):
        num_joints = p.getNumJoints(self.robot)

        joint_min_limits = []
        joint_max_limits = []

        for joint in range(num_joints):
            joint_info = p.getJointInfo(self.robot, joint)
            joint_type = joint_info[2]
            if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
                joint_min_limit = joint_info[8]
                joint_max_limit = joint_info[9]
                joint_min_limits.append(joint_min_limit)
                joint_max_limits.append(joint_max_limit)

        return joint_min_limits, joint_max_limits

    def _is_grasped(self):
        return all(
            len(p.getContactPoints(bodyA=self.robot, 
                                   bodyB=self.cube,
                                   linkIndexA=link)) > 0
            for link in self.active_joints[7:]
        )


if __name__ == "__main__":
    env = RobotEnvironment("human")
    env.step([0.5, 0.5, 0.5, 1.0])
    time.sleep(5)
    observation, info = env.reset()


    while True:
        p.stepSimulation()