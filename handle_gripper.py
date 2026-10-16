import pybullet as p
import pybullet_data
import time

ROBOT_HOME_POSITION = [0.002434193097806552, -0.7811217832664136, 0.009772560588829905, -2.062867921360752, 
                       -0.007282385107511599, 1.4879767515924243, 0.8049546850317025, 0.0]
RANDOM_POSE = [0.0, -0.35, 0.0, -1.85, 0.0, 1.5, 0.785, 0.0]
REVOLUTE_JOINTS = [0, 1, 2, 3, 4, 5, 6]
PRISMATIC_JOINTS = [9, 10]
END_EFFECTOR_INDEX = 11
ROBOT_FINGER_OPEN = 0.04
ROBOT_FINGER_CLOSED = 0

def createEnvironment():
    # plane
    plane = p.loadURDF("plane.urdf")

    # table
    tablePosition = [0, 0, 0]
    tableOrientation = p.getQuaternionFromEuler([0, 0, 0])
    table = p.loadURDF("table/table.urdf", tablePosition, tableOrientation)

    table_min, table_max = p.getAABB(table, -1)

    # franka panda
    robotPosition = [table_min[0] / 2.0, 0, table_max[2]]
    robotOrientation = p.getQuaternionFromEuler([0, 0, 0])
    robot = p.loadURDF("franka_panda/panda.urdf", robotPosition, robotOrientation, useFixedBase=True)

    # target
    target_dimensions = [0.1, 0.1, 0.01]
    target_color = [1, 1, 0, 0.8]
    target_position = [table_max[0] / 4, 0, table_max[2]]
    target_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=target_dimensions, rgbaColor=target_color)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=target_visual, basePosition=target_position)

    # box
    box_dimensions = [0.025, 0.025, 0.025]
    box_color = [1, 0, 0, 0.8]
    box_position = [table_max[0] / 4, table_max[1] / 2, table_max[2]]
    box_collision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=box_dimensions)
    box_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=box_dimensions, rgbaColor=box_color)
    p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=box_collision, baseVisualShapeIndex=box_visual, basePosition=box_position)

    return robot, target, box


if __name__ == "__main__":
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    robot, target, box = createEnvironment()

    while True:
        p.stepSimulation()