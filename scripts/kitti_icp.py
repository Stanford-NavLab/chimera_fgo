import os
import numpy as np
from scipy.spatial.transform import Rotation as R

from chimera_fgo.util.general import lla_to_ecef, ecef2enu
from chimera_fgo.util.io import read_lidar_bin, read_gt
from chimera_fgo.registration import initialize_source_and_target, p2pl_ICP_with_covariance

# ----------------------- Parameters ----------------------- #
kitti_seq = '2011_10_03_drive_0034_sync'
start_idx = 0
ds_rate = 10
Q_ini_scale = 0.01

# ----------------------- Check if output data path exists ----------------------- #
data_path = os.path.join(os.getcwd(), '..', 'data', 'kitti', kitti_seq, 'results', 'p2pl_icp')
if not os.path.exists(data_path):
    print("Output data path does not exist!")

# ----------------------- Read lidar point clouds ----------------------- #
print("Loading LiDAR data...")
binpath = os.path.join(os.getcwd(), '..', 'data', 'kitti', kitti_seq, 'velodyne_points', 'data')
PC_data_all = read_lidar_bin(binpath)

# ----------------------- Select data ----------------------- #
PC_data = PC_data_all[start_idx:]

# ----------------------- Downsample points ----------------------- #
PC_data = [pc[::ds_rate] for pc in PC_data]

# ----------------------- Read ground-truth data ----------------------- #
print("Loading ground truth data...")
gtpath = os.path.join(os.getcwd(), '..', 'data', 'kitti', kitti_seq, 'oxts', 'data')
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

# ----------------------- Get initial heading ----------------------- #
heading = gt_data[0][5] # heading angle
r = R.from_euler('XYZ', [0, 0, heading])
R_heading = r.as_matrix()

# ----------------------- Run ICP ----------------------- #
N = len(PC_data)
R_abs = R_heading
t_abs = gt_ecef[0].copy()
poses = N * [None]
poses[0] = (R_abs.copy(), t_abs.copy())

lidar_Rs = []
lidar_ts = []
lidar_covariances = []

for i in range(1,N):
    print(i, "/", N)
    trans_init = np.eye(4)
    threshold = 1.0
    source, target = initialize_source_and_target(PC_data[i], PC_data[i-1])
    reg_p2p, covariance = p2pl_ICP_with_covariance(source, target, threshold, trans_init, Q_ini=Q_ini_scale*np.eye(6))
    R_hat = reg_p2p.transformation[:3,:3]
    t_hat = reg_p2p.transformation[:3,3]

    #print(covariance)

    lidar_Rs.append(R_hat)
    lidar_ts.append(t_hat)
    lidar_covariances.append(covariance)

    t_abs += (R_abs @ t_hat).flatten()
    R_abs = R_hat @ R_abs
    poses[i] = (R_abs.copy(), t_abs.copy())

positions = np.zeros((N,3))
for i in range(N):
    positions[i] = poses[i][1]

# ----------------------- Save registration results to file ----------------------- #
run_name = 'start_{}_ds_{}_Q_ini_{}'.format(start_idx, ds_rate, Q_ini_scale)
np.save(os.path.join(data_path, 'lidar_Rs_'+run_name+'.npy'), np.array(lidar_Rs))
np.save(os.path.join(data_path, 'lidar_ts_'+run_name+'.npy'), np.array(lidar_ts))
np.save(os.path.join(data_path, 'positions_'+run_name+'.npy'), positions)
np.save(os.path.join(data_path, 'covariances_'+run_name+'.npy'), np.array(lidar_covariances))