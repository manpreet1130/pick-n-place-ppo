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
    target = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=target_visual, basePosition=target_position)

    # box
    box_dimensions = [0.025, 0.025, 0.025]
    box_color = [1, 0, 0, 0.8]
    box_position = [table_max[0] / 4, table_max[1] / 2, table_max[2]]
    box_collision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=box_dimensions)
    box_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=box_dimensions, rgbaColor=box_color)
    box = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=box_collision, baseVisualShapeIndex=box_visual, basePosition=box_position)

    return robot, target, box

def moveGripperToJointPosition(robot, joints, target_positions, max_velocity=0.6):
    for joint, target in zip(joints, target_positions):
        p.setJointMotorControl2(robot, 
                                joint, 
                                controlMode=p.POSITION_CONTROL, 
                                targetPosition=target, 
                                maxVelocity=max_velocity)
    return executeAction(robot, joints, target_positions)

def executeAction(robot, joints, target_positions, tolerance=1e-3, max_steps=1000):
    for step in range(max_steps):
        p.stepSimulation()
        time.sleep(1.0 / 240)
        errors = [abs(p.getJointState(robot, joint)[0] - target) for joint, target in zip(joints, target_positions)]
        if max(errors) <= tolerance:
            return True
    print("could not reach the desired configuration...")
    return False

def openCloseGripper(robot, open = False, tolerance=1e-3, position_tolerance=1e-5, force=40, max_steps=250):
    gripperTarget = ROBOT_FINGER_OPEN if open else ROBOT_FINGER_CLOSED
    for joint in PRISMATIC_JOINTS:
        p.setJointMotorControl2(robot, 
                                joint,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=gripperTarget,
                                force=force)

    previousPositions = None
    for step in range(max_steps):
        p.stepSimulation()
        time.sleep(1.0 / 240)

        positions = [p.getJointState(robot, joint)[0] for joint in PRISMATIC_JOINTS]

        if all(abs(positions[joint] - gripperTarget) < tolerance for joint in range(len(positions))):
            return True

        stallCount = 0
        if previousPositions and all(abs(positions[joint] - previousPositions[joint]) < position_tolerance for joint in range(len(positions))):
            stallCount += 1

            if stallCount > 5:
                return not open
        else:
            stallCount = 0
        previousPositions = positions
    print("max steps reached but gripper didnt open/close")
    return False

if __name__ == "__main__":
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    robot, target, box = createEnvironment()

    # moveGripperToJointPosition(robot, REVOLUTE_JOINTS, RANDOM_POSE)
    # openCloseGripper(robot, True)
    # moveGripperToJointPosition(robot, REVOLUTE_JOINTS, ROBOT_HOME_POSITION)
    # openCloseGripper(robot, False)

    cubePosition, cubeOrientation = p.getBasePositionAndOrientation(box)
    print(cubePosition, cubeOrientation)


    while True:
        p.stepSimulation()