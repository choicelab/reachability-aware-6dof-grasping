from coppelia.sim_lib import SimEnv, SimCamera, SimRobot
from pointcloud import PointCloud
from sample import GraspPoseSampler
import coppelia.sim as vrep
import time
import numpy as np
import torch
from model import CNN3d
from utils import voxelize, rotate_cloud
import argparse
import params

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(args):
    # 3D CNN grasp planner
    grasp_model = CNN3d().to(device)
    grasp_model.load_state_dict(torch.load('weights/gsp_pretrained.pt'))
    grasp_model.eval()

    env = SimEnv()
    robot = SimRobot(env.cid)
    sim_cam = SimCamera(env.cid, 'camera_view1')
    while True:
        env.add_object("object_0", [0, 0, 0.1], random_angle=True)
        time.sleep(0.5)
        raw_color, raw_depth = sim_cam.read_buffer()
        raw_pcd = PointCloud(raw_color, raw_depth, sim_cam.intrinsics)
        raw_pcd.make_pointcloud(extrinsics=sim_cam.extrinsics)
        # raw_pcd.view_point_cloud()
        raw_pcd_npy = np.array(raw_pcd._o3d_pc.points)
        print("Point cloud with {} points".format(raw_pcd_npy.shape[0]))
        if raw_pcd_npy.shape[0] < 100:
            continue
        sampler = GraspPoseSampler(nb_grasps=200, high=45, pc=raw_pcd_npy)
        pose_set = sampler.sample()
        X1 = []
        for i, grasp_pose in enumerate(pose_set, 0):
            P1 = rotate_cloud(grasp_pose[:, 3], grasp_pose[:, :3], raw_pcd_npy)
            v1 = voxelize(P1, args.vg_size, 40)
            X1.append(v1)
        X1 = np.array(X1)
        with torch.no_grad():
            output = grasp_model(torch.tensor(X1, dtype=torch.float).to(device).unsqueeze(1)).squeeze().cpu().numpy()
        final_idx = np.argmax(np.array(output))
        a = pose_set[final_idx]
        label = robot.grasp(a[:, 3], a[:, :3], "object_0")
        print("grasping result: {}".format(label))

        vrep.simxStopSimulation(env.cid, vrep.simx_opmode_blocking)
        time.sleep(2.0)
        vrep.simxStartSimulation(env.cid, vrep.simx_opmode_blocking)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()

    # --------------- Setup options ---------------
    parser.add_argument('--data_dir', dest='data_dir', type=str, default="/home/lou00015/ReachabilityAwareGrasping/data",
                        help='directory to where grasping data is saved')
    parser.add_argument('--vg_size', dest='vg_size', type=float, default=params.VOXEL_GRID_SIZE,
                        help='size of the voxel grid in meters')
    # Process data with specified arguments
    args = parser.parse_args()
    test(args)
