import sys
sys.path.append('/home/lou00015/ReachabilityAwareGrasping')
from coppelia.sim_lib import SimEnv, SimCamera, SimRobot
from pointcloud import PointCloud
from sample import GraspPoseSampler
import coppelia.sim as vrep
import time
import numpy as np
import os.path as osp
import argparse


def generate_data(args):
    env = SimEnv()
    robot = SimRobot(env.cid)
    sim_cam = SimCamera(env.cid, 'camera_view1')
    i = args.starting_index
    object_name = args.object_name
    data_dir = args.data_dir
    while i < args.nb_data:
        env.add_object(object_name, [0, 0, 0.25], random_angle=True)
        time.sleep(2.0)
        raw_color, raw_depth = sim_cam.read_buffer()
        raw_pcd = PointCloud(raw_color, raw_depth, sim_cam.intrinsics)
        raw_pcd.make_pointcloud(extrinsics=sim_cam.extrinsics)
        # raw_pcd.view_point_cloud()
        raw_pcd_npy = np.array(raw_pcd._o3d_pc.points)
        print("Point cloud with {} points".format(raw_pcd_npy.shape[0]))
        if raw_pcd_npy.shape[0] < 100:
            vrep.simxStopSimulation(env.cid, vrep.simx_opmode_blocking)
            time.sleep(2.0)
            vrep.simxStartSimulation(env.cid, vrep.simx_opmode_blocking)
            continue
        sampler = GraspPoseSampler(nb_grasps=50, high=45, pc=raw_pcd_npy)
        pose_set = sampler.sample()
        pose = pose_set[25]
        print("pose is:", pose)
        np.save(osp.join(data_dir, 'pc_{}.npy'.format(i)), raw_pcd_npy)
        np.save(osp.join(data_dir, 'g_{}.npy'.format(i)), pose)
        label = robot.grasp(pose[:, 3], pose[:, :3], object_name)
        print("grasping result: {}".format(label))
        np.save(osp.join(data_dir, 'l_{}.npy'.format(i)), label)
        vrep.simxStopSimulation(env.cid, vrep.simx_opmode_blocking)
        time.sleep(2.0)
        vrep.simxStartSimulation(env.cid, vrep.simx_opmode_blocking)
        i += 1


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()

    # --------------- Setup options ---------------
    parser.add_argument('--data_dir', dest='data_dir', type=str, default="/home/lou00015/data/grasping",
                        help='directory to where grasping data is saved')
    parser.add_argument('--object_name', dest='object_name', type=str, default="object_0",
                        help='object being grasped')
    parser.add_argument('--nb_data', dest='nb_data', type=int, default=60000,
                        help='number of data that will be collected')
    parser.add_argument('--starting_index', dest='starting_index', type=int, default=0,
                        help='starting index of the data')
    # Process data with specified arguments
    data_args = parser.parse_args()
    generate_data(data_args)
