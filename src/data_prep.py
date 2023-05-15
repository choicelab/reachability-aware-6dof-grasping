import numpy as np
import os
from utils import voxelize, rotate_cloud
import matplotlib.pyplot as plt
import sys
import argparse


def grasping_data(args):
    x = []
    y = []
    for idx in range(args.nb_data):
        pc = np.load(os.path.join(args.data_dir, "pc_{}.npy".format(idx)))
        l = np.load(os.path.join(args.data_dir, "l_{}.npy".format(idx)))
        g = np.load(os.path.join(args.data_dir, "g_{}.npy".format(idx)))

        pc_g = rotate_cloud(g[:, 3], g[:, :3], pc)
        vg = voxelize(pc_g, 0.2, 40)
        # print(l)


        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.voxels(vg, facecolors='green', edgecolors='k')
        # plt.show()

        x.append(vg)
        y.append(l)
        sys.stdout.write("\r Processing {}/{}".format(idx, args.nb_data))
        sys.stdout.flush()
    print("Success rate of the dataset: {}".format(sum(y)/len(y)))
    print(sum(y))
    np.savez_compressed('data/grasping.npz', x=x, y=y)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()

    # --------------- Setup options ---------------
    parser.add_argument('--data_dir', dest='data_dir', type=str, default="/home/lou00015/data/grasping",
                        help='directory to where grasping data is saved')
    parser.add_argument('--nb_data', dest='nb_data', type=int, default=60000,
                        help='number of data that will be processed')
    # Process data with specified arguments
    args = parser.parse_args()
    grasping_data(args)
