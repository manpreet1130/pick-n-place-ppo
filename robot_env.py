import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np

class RobotEnvironment(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30
    }

    def __init__(self, render_mode=None):
        super().__init__()

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
        self.table, self.robot, self.cube = self.loadScene()
        self._lower_limits, self._upper_limits = self._getLimits()
                
        # initialize observation space and action space
        self.observation_space = spaces.Box(
            low=np.array(self._lower_limits, dtype=np.float32),
            high=np.array(self._upper_limits, dtype=np.float32),
            dtype=np.float32
        )

        while True:
            p.stepSimulation()


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
        target_half_dimensions = [0.1, 0.1, 0.01]
        target_color = [1, 1, 0, 0.8]
        target_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=target_half_dimensions, rgbaColor=target_color)
        target_position, target_orientation = [table_max[0] / 4.0, 0, table_max[2]], p.getQuaternionFromEuler([0, 0, 0])
        target = p.createMultiBody(baseMass=0, 
                                   baseCollisionShapeIndex=-1, 
                                   baseVisualShapeIndex=target_visual,
                                   basePosition=target_position,
                                   baseOrientation=target_orientation)

        # cube
        cube_half_dimensions = [0.030, 0.030, 0.030]
        cube_color = [1, 0, 0, 0.8]
        cube_collision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=cube_half_dimensions)
        cube_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=cube_half_dimensions, rgbaColor=cube_color)
        cube_position, cube_orientation = [table_max[0] / 4.0, table_max[1] / 4.0, table_max[2]], p.getQuaternionFromEuler([0, 0, 0])
        cube = p.createMultiBody(baseMass=0.5, 
                                 baseCollisionShapeIndex=cube_collision,
                                 baseVisualShapeIndex=cube_visual,
                                 basePosition=cube_position,
                                 baseOrientation=cube_orientation)    
        return table, robot, cube

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

        table_min, table_max = p.getAABB(self.table, -1)
        # [r1, r2, r3, r4, r5, r6, r7, p1, p2, rx, ry, rz, cx, cy, cz, tx, ty, tz]
        joint_min_limits = [*joint_min_limits, *table_min, *table_min, *table_min]
        joint_max_limits = [*joint_max_limits, *table_max, *table_max, *table_max]

        return joint_min_limits, joint_max_limits


if __name__ == "__main__":
    env = RobotEnvironment("human")
