import random
import time
import numpy as np
import coppelia.sim as vrep
from src.pointcloud import PointCloud
from src.sample import GraspPoseSampler


class SimCamera(object):
    def __init__(self, clientID, camera_name, depth_only=False):
        self.cid = clientID
        self.rgb_handle = None
        if not depth_only:
            sim_ret, cam_rgb_handle = vrep.simxGetObjectHandle(self.cid, camera_name + '_color',
                                                               vrep.simx_opmode_blocking)
            self.rgb_handle = cam_rgb_handle
        sim_ret, cam_depth_handle = vrep.simxGetObjectHandle(self.cid, camera_name + '_depth',
                                                             vrep.simx_opmode_blocking)
        self.depth_handle = cam_depth_handle
        res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(self.cid, camera_name + '_depth',
                                                                                     vrep.sim_scripttype_childscript,
                                                                                     'getMatrix', [], [], [],
                                                                                     bytearray(),
                                                                                     vrep.simx_opmode_blocking)
        self.intrinsics = np.asarray([[589.366454183, 0, 320], [0, 589.366454183, 240], [0, 0, 1]])

        self.depth_m = np.asarray([[retFloats[0], retFloats[1], retFloats[2], retFloats[3]],
                                   [retFloats[4], retFloats[5], retFloats[6], retFloats[7]],
                                   [retFloats[8], retFloats[9], retFloats[10], retFloats[11]]])
        self.extrinsics = np.asarray([[retFloats[0], retFloats[1], retFloats[2], retFloats[3]],
                                   [retFloats[4], retFloats[5], retFloats[6], retFloats[7]],
                                   [retFloats[8], retFloats[9], retFloats[10], retFloats[11]],
                                   [0, 0, 0, 1]])

    def read_buffer(self):
        # Get depth image from simulation
        sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.cid, self.depth_handle,
                                                                                vrep.simx_opmode_blocking)
        depth_img = np.asarray(depth_buffer, dtype=np.float64)
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = np.fliplr(depth_img)
        zNear = 0.01
        zFar = 3.5
        depth_img = depth_img * (zFar - zNear) + zNear
        if self.rgb_handle:
            # Get color image from simulation
            sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.cid, self.rgb_handle, 0,
                                                                           vrep.simx_opmode_blocking)
            color_img = np.asarray(raw_image)
            color_img.shape = (resolution[1], resolution[0], 3)
            color_img = color_img.astype(float) / 255
            color_img[color_img < 0] += 1
            color_img *= 255
            color_img = np.fliplr(color_img)
            color_img = color_img.astype(np.uint8)
            return color_img, depth_img
        else:
            return depth_img


class SimEnv(object):
    def __init__(self):
        # self.wd = params.ROOT_DIR
        self.cid = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
        vrep.simxStartSimulation(self.cid, vrep.simx_opmode_blocking)

    def add_object(self, object_name, object_pos, random_angle=False) -> tuple:
        res, object_handle = vrep.simxGetObjectHandle(self.cid, object_name, vrep.simx_opmode_oneshot_wait)
        if not random_angle:
            print('Adding {} from the ycb set'.format(object_name))
            res, abg = vrep.simxGetObjectOrientation(self.cid, object_handle, -1, vrep.simx_opmode_oneshot_wait)
            object_angle = [abg[0], abg[1], random.uniform(-np.pi, np.pi)]
            # random.uniform(-np.pi, np.pi)
        else:
            object_angle = [random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi)]
        vrep.simxSetObjectOrientation(self.cid, object_handle, -1, object_angle, vrep.simx_opmode_oneshot)
        vrep.simxSetObjectPosition(self.cid, object_handle, -1, object_pos, vrep.simx_opmode_oneshot)
        time.sleep(1.0)
        return object_name, object_handle

    def setup_cams(self, cam_names) -> dict:
        cam_dict = {}
        for c_name in cam_names:
            cam_dict[c_name] = SimCamera(clientID=self.cid, camera_name=c_name, depth_only=False)
        return cam_dict


class SimRobot(object):
    def __init__(self, cid):
        self.cid = cid

    def get_object_orientation(self, object_name):
        _, object_handle = vrep.simxGetObjectHandle(self.cid, object_name, vrep.simx_opmode_blocking)
        # _, xyz = vrep.simxGetObjectPosition(self.cid, object_handle, -1, vrep.simx_opmode_blocking)
        _, abg = vrep.simxGetObjectOrientation(self.cid, object_handle, -1, vrep.simx_opmode_blocking)
        return abg

    def get_object_position(self, object_name):
        _, object_handle = vrep.simxGetObjectHandle(self.cid, object_name, vrep.simx_opmode_blocking)
        _, xyz = vrep.simxGetObjectPosition(self.cid, object_handle, -1, vrep.simx_opmode_blocking)
        # _, abg = vrep.simxGetObjectOrientation(self.cid, object_handle, -1, vrep.simx_opmode_blocking)
        return xyz

    def grasp(self, pt, pose, object_name=None):
        goal1 = [pose[0][0], pose[0][1], pose[0][2], pt[0],
                 pose[1][0], pose[1][1], pose[1][2], pt[1],
                 pose[2][0], pose[2][1], pose[2][2], pt[2]]
        vrep.simxCallScriptFunction(self.cid, 'target', vrep.sim_scripttype_childscript, 'setGoal', [],
                                    goal1, [], bytearray(), vrep.simx_opmode_blocking)
        goal2 = [pose[0][0], pose[0][1], pose[0][2], pt[0],
                 pose[1][0], pose[1][1], pose[1][2], pt[1],
                 pose[2][0], pose[2][1], pose[2][2], pt[2] + 0.3]
        vrep.simxCallScriptFunction(self.cid, 'ending', vrep.sim_scripttype_childscript, 'setending', [],
                                    goal2, [], bytearray(), vrep.simx_opmode_blocking)
        time.sleep(1.0)
        vrep.simxCallScriptFunction(self.cid, 'Sphere', vrep.sim_scripttype_childscript, 'grasp', [], [],
                                    [], bytearray(), vrep.simx_opmode_blocking)
        while True:
            res, finish = vrep.simxGetIntegerSignal(self.cid, "finish", vrep.simx_opmode_oneshot_wait)
            if finish == 1:
                break
        c = [0, 0, 0]
        if object_name:
            c = self.get_object_position(object_name)
        if c[2] > 0.1:
            return 1
        else:
            return 0

    def place(self, pt, pose):
        goal1 = [pose[0][0], pose[0][1], pose[0][2], pt[0],
                 pose[1][0], pose[1][1], pose[1][2], pt[1],
                 pose[2][0], pose[2][1], pose[2][2], pt[2]]
        vrep.simxCallScriptFunction(self.cid, 'target', vrep.sim_scripttype_childscript, 'setGoal', [],
                                    goal1, [], bytearray(), vrep.simx_opmode_blocking)
        goal2 = [pose[0][0], pose[0][1], pose[0][2], pt[0],
                 pose[1][0], pose[1][1], pose[1][2], pt[1],
                 pose[2][0], pose[2][1], pose[2][2], pt[2] - 0.2]
        vrep.simxCallScriptFunction(self.cid, 'ending', vrep.sim_scripttype_childscript, 'setending', [],
                                    goal2, [], bytearray(), vrep.simx_opmode_blocking)
        time.sleep(1.0)
        vrep.simxCallScriptFunction(self.cid, 'Sphere', vrep.sim_scripttype_childscript, 'place', [], [],
                                    [], bytearray(), vrep.simx_opmode_blocking)
        while True:
            res, finish = vrep.simxGetIntegerSignal(self.cid, "finish", vrep.simx_opmode_oneshot_wait)
            if finish == 1:
                break
        return 1

    def push(self, start, goal):
        start[2] += 0.01
        goal[2] += 0.01
        eoat_x_axis = (goal - start) / np.linalg.norm(goal - start)
        eoat_y_axis = np.cross(np.array([0, 0, -1]), eoat_x_axis)
        goal1 = [eoat_x_axis[0], eoat_y_axis[0], 0, start[0] - eoat_x_axis[0]*0.01,
                 eoat_x_axis[1], eoat_y_axis[1], 0, start[1] - eoat_x_axis[1]*0.01,
                 eoat_x_axis[2], eoat_y_axis[2], -1, start[2]]
        vrep.simxCallScriptFunction(self.cid, 'target', vrep.sim_scripttype_childscript, 'setGoal', [],
                                    goal1, [], bytearray(), vrep.simx_opmode_blocking)
        goal2 = [eoat_x_axis[0], eoat_y_axis[0], 0, goal[0],
                 eoat_x_axis[1], eoat_y_axis[1], 0, goal[1],
                 eoat_x_axis[2], eoat_y_axis[2], -1, goal[2]]
        vrep.simxCallScriptFunction(self.cid, 'ending', vrep.sim_scripttype_childscript, 'setending', [],
                                    goal2, [], bytearray(), vrep.simx_opmode_blocking)
        time.sleep(1.0)
        vrep.simxCallScriptFunction(self.cid, 'Sphere', vrep.sim_scripttype_childscript, 'push', [], [],
                                    [], bytearray(), vrep.simx_opmode_blocking)
        while True:
            res, finish = vrep.simxGetIntegerSignal(self.cid, "finish", vrep.simx_opmode_oneshot_wait)
            if finish == 1:
                break
        return 1


