import numpy as np
import open3d as o3d
import os
from scipy.spatial.transform import Rotation as R

from chimera_fgo.general import lla_to_ecef, ecef2enu
from chimera_fgo.io import read_lidar_bin, read_gt
from chimera_fgo.registration import initialize_source_and_target, p2pl_ICP_with_covariance

if __name__=='__main__':
    # Load point clouds
    binpath = 'oak/stanford/groups/gracegao/kitti_raw/2011_09_30/2011_09_30_drive_0028_sync/velodyne_points/data'
    #binpath = os.path.join(os.getcwd(), '..', 'data', 'kitti', '2011_09_30_drive_0028_sync', 'velodyne_points', 'data')
    PC_data_all = read_lidar_bin(binpath)

    start_idx = 1550
    PC_data = PC_data_all[start_idx:]

    # Load ground truth
    gtpath = os.path.join(os.getcwd(), '..', 'data', 'kitti', '2011_09_30_drive_0028_sync', 'oxts', 'data')
    gt_data = read_gt(gtpath)
    gt_data = gt_data[start_idx:]
    lla = gt_data[:,:3]    

    ref_lla = lla[0]
    ecef = lla_to_ecef(*lla[0])
    gt_ecef = np.zeros((len(lla),3))

    for i in range(len(lla)):
        ecef = lla_to_ecef(*lla[i])
        gt_ecef[i] = ecef2enu(ecef[0], ecef[1], ecef[2], ref_lla[0], ref_lla[1], ref_lla[2])

    gt_ecef = gt_ecef[:,[1,0,2]]

    # Get initial orientation
    heading = gt_data[0][5] # heading angle
    r = R.from_euler('XYZ', [0, 0, heading])
    R_heading = r.as_matrix()

    # Run ICP
    N = len(PC_data)
    #N = 1000
    R_abs = R_heading
    t_abs = gt_ecef[0].copy()
    poses = N * [None]
    poses[0] = (R_abs.copy(), t_abs.copy())

    lidar_Rs = []
    lidar_ts = []
    covariances = []

    for i in range(1,N):
        print(i, "/", N)
        trans_init = np.eye(4)
        threshold = 1
        source, target = initialize_source_and_target(PC_data[i], PC_data[i-1])
        reg_p2pl, covariance = p2pl_ICP_with_covariance(source, target, threshold, trans_init)
        R_hat = reg_p2pl.transformation[:3,:3]
        t_hat = reg_p2pl.transformation[:3,3]

        lidar_Rs.append(R_hat)
        lidar_ts.append(t_hat)
        covariances.append(covariance)

