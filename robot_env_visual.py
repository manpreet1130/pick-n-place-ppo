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
        assert render_mode == None or render_mode in self.metadata.get("render_modes")
        self.render_mode = render_mode
        if self.render_mode == "human":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.target_half_extents = [0.1, 0.1, 0.01]
        self.target_colour = [1, 1, 0, 1]
        self.cube_length = 0.04
        self.cube_half_extents = [self.cube_length, self.cube_length, self.cube_length]
        self.cube_colour = [1, 0, 0, 1]
        self.max_steps_per_episode = 400

        self.robot, self.cube, self.target = self.createScene()

        self.minimum_lift_height = 0.05
        self.has_lifted = False
        self.grip_strength = 5 # 5 N grip strength from each gripper to the cube
        self.body_joints = [0, 1, 2, 3, 4, 5, 6]
        self.gripper_joints = [9, 10]
        self.gripper_index = 11
        self.distance_target_cube_threshold = 0.07 # 7 cm gap between target center and cube center
        self.camera_height, self.camera_width = 128, 128

        proprio_length = len(self.body_joints) + len(self.gripper_joints) + 1 + 1 + 3
        self.observation_space = spaces.Dict({
            "camera_image": spaces.Box(
                low=0,
                high=255,
                shape=(self.camera_height, self.camera_width, 3),
                dtype=np.uint8
            ),
            "proprio": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(proprio_length,),
                dtype=np.float32
            )
        })

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
            aspect=self.camera_width / self.camera_height,
            nearVal=0.3,
            farVal=3.0
        )

    def createScene(self):
        p.loadURDF("plane.urdf")

        robotPosition, robotOrientation = [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0])
        robot = p.loadURDF("franka_panda/panda.urdf", robotPosition, robotOrientation, useFixedBase=True)

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

        # generating a random cube position between the 0.33 - 0.65 (comfort range of the robot) 
        while True:
            theta = self.np_random.uniform(-np.pi/2.0, np.pi/2.0)
            distance = np.sqrt(self.np_random.uniform(0.33 ** 2, 0.65 ** 2))
            x = robotPosition[0] + distance * math.cos(theta)
            y = robotPosition[1] + distance * math.sin(theta)
            z = robotPosition[2] + self.cube_half_extents[2]
            distance_target_cube = np.linalg.norm(np.array([x, y], dtype=np.float32) - np.array(targetPosition, dtype=np.float32)[:2])
            if distance_target_cube > 0.15:
                break

        cubePosition, cubeOrientation = [x, y, z], p.getQuaternionFromEuler([0, 0, 0])
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
                            basePosition=cubePosition,
                            baseOrientation=cubeOrientation)

        return robot, cube, target

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
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix
        )

        image = np.asarray(rgb_image, dtype=np.uint8)
        image = image.reshape(self.camera_height, self.camera_width, 4)
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