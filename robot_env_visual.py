'''
    - main objective of this is to use a camera image to be able to pick
    and place a cube at a desired target using the franka panda arm
    - extension to this is to use the same to stack cubes (3 maybe)


    TODO:
        - init
            - observation space
            - action space
            - camera calibration
            - distance threshold
            - max step size per episode
        - information
            - cube position
            - target position
            - distance between gripper and cube
            - distance between cube and target (in xy plane)
            - is_grasped
            - is_success
            - has_lifted
            - cube height 
        - observation
            - rgb image
            - proprio: 
                - joint angles (revolute + prismatic)
                - has gripped
        - reset
            - bring the robot to the home position
            - initialize cube in a random position 
        - step
            - inverse kinematics to next position
            - reward calculation
            - terminated -> distance between cube and target < threshold
            - truncated -> max step size reached
            - return observation, reward, terminated, truncated, information
        - render
        - close
'''
import pybullet as p
import gymnasium as gym
from gymnasium import spaces
import pybullet_data
import time
import numpy as np
import math


class RobotEnvironmentVisual(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30
    }

    def __init__(self, render_mode=None):
        super().__init__()
        assert render_mode is None or render_mode in self.metadata.get("render_modes")
        self.render_mode = render_mode
        if self.render_mode == "human":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        self.home_position = [0.002434193097806552, -0.7811217832664136, 0.009772560588829905, -2.062867921360752, 
                       -0.007282385107511599, 1.4879767515924243, 0.8049546850317025, 0.04, 0.04]

        self.target_half_extents = [0.1, 0.1, 0.01]
        self.target_colour = [1, 1, 0, 1]
        self.cube_length = 0.03
        self.cube_half_extents = [self.cube_length, self.cube_length, self.cube_length]
        self.cube_colour = [1, 0, 0, 1]
        self.max_steps_per_episode = 400
        self.current_step = 0

        self.robot, self.cube, self.target = self.create_scene()

        self.minimum_lift_height = 0.05 + self.cube_length
        self.has_lifted = False
        self.grip_strength = 5 # 5 N grip strength from each gripper to the cube
        self.body_joints = [0, 1, 2, 3, 4, 5, 6]
        self.gripper_joints = [9, 10]
        self.gripper_index = 11
        self.distance_target_cube_threshold = 0.07 # 7 cm gap between target center and cube center
        self.render_width, self.render_height = 640, 480
        self.observation_width, self.observation_height = 128, 128
        self.gripper_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])
        self.gripper_open = [0.04, 0.04]
        self.gripper_closed = [0.0, 0.0]
        self.gripper_force = [100.0, 100.0]

        self.active_joints = np.concatenate([self.body_joints, self.gripper_joints])
        self.joint_lower_limits, self.joint_upper_limits = self._get_joint_limits(self.active_joints)
        self.ranges = [upper - lower for upper, lower in zip(self.joint_upper_limits, self.joint_lower_limits)]

        self.action_scale = 0.03

        proprio_length = len(self.body_joints) + len(self.gripper_joints) + 1 + 1 + 3
        self.observation_space = spaces.Dict({
            "camera_image": spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8
            ),
            "proprio": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(proprio_length,),
                dtype=np.float32
            )
        })

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(4,),
            dtype=np.float32
        )

        # camera setup
        target_position, _ = p.getBasePositionAndOrientation(self.target)
        camera_target_position = [target_position[0] / 2.0, target_position[1] / 2.0, target_position[2] + 0.2]
        
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_target_position,
            distance=1.5,
            yaw=90,
            pitch=-50,
            roll=0,
            upAxisIndex=2
        )

        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.observation_width / self.observation_height,
            nearVal=0.3,
            farVal=3.0
        )

        self.render_view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_target_position,
            distance=1.6,
            yaw=45,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )

        self.render_projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.render_width / self.render_height,
            nearVal=0.1,
            farVal=5.0
        )


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.has_lifted = False

        for i, joint in enumerate(self.active_joints):
            p.resetJointState(
                bodyUniqueId=self.robot, 
                jointIndex=joint,
                targetValue=self.home_position[i]
            )

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.active_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=self.home_position
        )

        robot_position, _ = p.getBasePositionAndOrientation(self.robot)
        target_position, _ = p.getBasePositionAndOrientation(self.target)
        cube_position = self._sample_cube_position(robot_position, target_position)

        p.resetBasePositionAndOrientation(self.cube, cube_position, p.getQuaternionFromEuler([0, 0, 0]))
        p.resetBaseVelocity(self.cube, [0, 0, 0], [0, 0, 0])

        for _ in range(50):
            p.stepSimulation()

        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def render(self):
        if self.render_mode == "rgb_array":
            _, _, rgb_image, _, _ = p.getCameraImage(
                    width=self.render_width,
                    height=self.render_height,
                    viewMatrix=self.render_view_matrix,
                    projectionMatrix=self.render_projection_matrix
            )
            image = np.asarray(rgb_image, dtype=np.uint8)
            image = image.reshape(self.render_height, self.render_width, 4)
            image = image[:, :, :3]
            return image
        return None

    def close(self):
        if p.isConnected():
            p.disconnect()

    def step(self, action):
        self.current_step += 1
        gripper_position = self._get_info().get("gripper_position")
        next_target_position = np.clip(gripper_position + action[:3] * self.action_scale, [0, -0.7, 0.02], [0.75, 0.7, 0.3])

        joint_angles = p.calculateInverseKinematics(
            bodyUniqueId=self.robot,
            endEffectorLinkIndex=self.gripper_index,
            targetPosition=next_target_position,
            targetOrientation=self.gripper_orientation,
            lowerLimits=self.joint_lower_limits,
            upperLimits=self.joint_upper_limits,
            restPoses=self.home_position,
            jointRanges=self.ranges,
            maxNumIterations=100
        )

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.body_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=joint_angles[:7]
        )

        if action[3] > 0:
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot,
                jointIndices=self.gripper_joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=self.gripper_closed,
                forces=self.gripper_force
            )
        else:
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot,
                jointIndices=self.gripper_joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=self.gripper_open,
                forces=self.gripper_force
            )

        for _ in range(30):
            p.stepSimulation()
            if self.render_mode == "human":
                time.sleep(1.0 / 240.0)

        cube_position, _ = p.getBasePositionAndOrientation(self.cube)
        target_position, _ = p.getBasePositionAndOrientation(self.target)

        if cube_position[2] >= self.minimum_lift_height:
            self.has_lifted = True

        distance_cube_target = np.linalg.norm(np.array(cube_position[:2], dtype=np.float32) - np.array(target_position[:2], dtype=np.float32))

        if distance_cube_target > self.distance_target_cube_threshold and cube_position[2] - self.cube_half_extents[2] <= 0.005 and not self._is_grasped():
            self.has_lifted = False
        

        observation = self._get_observation()
        info = self._get_info()
        reward = float(self._calculate_reward(info))
        terminated = info.get("is_success")
        truncated = self.current_step >= self.max_steps_per_episode

        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, info):
        '''
            stage 1:
                - currently the gripper is away from the cube, we want to go to the cube and grab
                - result: grab the cube
            stage 2:
                - cube grabbed, now its time to lift it
                - result: lifted to a minimum height defined
            stage 3: 
                - cube lifted to a certain height, now we need to go to the target
                - result: getting close to the target
            stage 4:
                - within a threshold distance from the target, time to drop the cube on the target
                - result: put down the cube in relation to the height we are currently at
            stage 5: 
                - cube on the target within a certain threshold distance
        '''
        if info.get("is_success"):
            return 100

        if info.get("has_lifted") and info.get("distance_cube_target") <= self.distance_target_cube_threshold:
            return -1 - 8 * (info.get("cube_height") - self.cube_half_extents[2])

        if info.get("has_lifted") and info.get("cube_height") >= self.minimum_lift_height:
            return -2 - info.get("distance_cube_target")

        if info.get("is_grasped"):
            return -3 - 15 * (self.minimum_lift_height - info.get("cube_height"))

        return -4 - info.get("distance_cube_gripper")
            

    def create_scene(self):
        p.loadURDF("plane.urdf")

        robot_position, robot_orientation = [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0])
        robot = p.loadURDF("franka_panda/panda.urdf", robot_position, robot_orientation, useFixedBase=True)

        targetPosition, targetOrientation = [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0])
        target_visual = p.createVisualShape(
                            shapeType=p.GEOM_BOX,
                            halfExtents=self.target_half_extents,
                            rgbaColor=self.target_colour)
        target = p.createMultiBody(
                            baseMass=0.0,
                            baseCollisionShapeIndex=-1,
                            baseVisualShapeIndex=target_visual,
                            basePosition=targetPosition,
                            baseOrientation=targetOrientation)

        cube_position, cube_orientation = self._sample_cube_position(robot_position, targetPosition), p.getQuaternionFromEuler([0, 0, 0])
        cube = self._generate_cube(cube_position, cube_orientation)
        return robot, cube, target

    def _generate_cube(self, position, orientation):
        cube_collision = p.createCollisionShape(
                            shapeType=p.GEOM_BOX,
                            halfExtents=self.cube_half_extents)
        cube_visual = p.createVisualShape(
                            shapeType=p.GEOM_BOX,
                            halfExtents=self.cube_half_extents,
                            rgbaColor=self.cube_colour)
        cube = p.createMultiBody(
                            baseMass=0.5,
                            baseCollisionShapeIndex=cube_collision,
                            baseVisualShapeIndex=cube_visual,
                            basePosition=position,
                            baseOrientation=orientation)
        return cube

    def _get_joint_limits(self, joints):
        joint_information = [p.getJointInfo(self.robot, joint) for joint in joints]
        lower_limits = [joint[8] for joint in joint_information]
        upper_limits = [joint[9] for joint in joint_information]
        return lower_limits, upper_limits

    def _is_grasped(self):
        contactsLeftFinger = p.getContactPoints(
                                    bodyA=self.robot,
                                    bodyB=self.cube,
                                    linkIndexA=self.gripper_joints[0])
        contactsRightFinger = p.getContactPoints(
                                    bodyA=self.robot,
                                    bodyB=self.cube,
                                    linkIndexA=self.gripper_joints[1])
        contactForceLeft = np.sum([point[9] for point in contactsLeftFinger])
        contactForceRight = np.sum([point[9] for point in contactsRightFinger])
        if contactForceLeft >= self.grip_strength and contactForceRight >= self.grip_strength:
            return True
        return False
    
    def _sample_cube_position(self, robot_position, targetPosition):
        # generating a random cube position between the 0.33 - 0.65 (comfort range of the robot) 
        while True:
            theta = self.np_random.uniform(-np.pi/2.0, np.pi/2.0)
            distance = np.sqrt(self.np_random.uniform(0.33 ** 2, 0.65 ** 2))
            x = robot_position[0] + distance * math.cos(theta)
            y = robot_position[1] + distance * math.sin(theta)
            z = robot_position[2] + self.cube_half_extents[2]
            distance_target_cube = np.linalg.norm(np.array([x, y], dtype=np.float32) - np.array(targetPosition, dtype=np.float32)[:2])
            if distance_target_cube > 0.15:
                break
        return np.array([x, y, z], dtype=np.float32)

    def _get_info(self):
        cube_position, _ = p.getBasePositionAndOrientation(self.cube)
        target_position, _ = p.getBasePositionAndOrientation(self.target)
        gripper_position = p.getLinkState(self.robot, self.gripper_index)[0]

        distance_cube_gripper = np.linalg.norm(np.array(cube_position, dtype=np.float32) - np.array(gripper_position, dtype=np.float32))
        distance_cube_target = np.linalg.norm(np.array(cube_position[:2], dtype=np.float32) - np.array(target_position[:2], dtype=np.float32))
        has_lifted = self.has_lifted
        is_grasped = self._is_grasped()
        cube_height = cube_position[2]
        is_success = (not is_grasped) and has_lifted and cube_height < self.cube_half_extents[2] + 0.01 \
                    and distance_cube_target < self.distance_target_cube_threshold

        return {
            "cube_position": cube_position,
            "cube_height": cube_height,
            "target_position": target_position,
            "gripper_position": gripper_position,
            "distance_cube_gripper": distance_cube_gripper,
            "distance_cube_target": distance_cube_target,
            "has_lifted": has_lifted,
            "is_grasped": is_grasped,
            "is_success": is_success
        }

    def _get_observation(self):
        '''
            observation space gets the following:
                - camera image
                - proprio:
                    - revolute joint angles,
                    - gripper joint positions,
                    - gripper position,
                    - is grasped,
                    - has lifted

        '''
        _, _, rgb_image, _, _ = p.getCameraImage(
            width=self.observation_width,
            height=self.observation_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix
        )

        image = np.asarray(rgb_image, dtype=np.uint8)
        image = image.reshape(self.observation_height, self.observation_width, 4)
        image = image[:, :, :3]

        body_joint_info = [p.getJointState(self.robot, joint)[0] for joint in self.body_joints]
        gripper_joint_info = [p.getJointState(self.robot, joint)[0] for joint in self.gripper_joints]
        info = self._get_info()
        gripper_position = info.get("gripper_position")
        is_grasped = [info.get("is_grasped")]
        has_lifted = [info.get("has_lifted")]

        proprio_info = np.concatenate([body_joint_info, gripper_joint_info, gripper_position, is_grasped, has_lifted], dtype=np.float32)

        return {
            "camera_image": image,
            "proprio": proprio_info
        }

if __name__ == "__main__":
    env = RobotEnvironmentVisual("human")

    while True:
        p.stepSimulation()
        env._get_observation()