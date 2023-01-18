import numpy as np
import struct 
import os
import plotly.graph_objects as go
import time
from scipy.stats import chi2
from tqdm import tqdm

import lgchimera.general as general
from lgchimera.io import read_lidar_bin, read_gt
from lgchimera.kitti_util import process_kitti_gt, load_icp_results
from lgchimera.geom_util import euler_to_R

from lgchimera.symforce.factor_graph import fgo

import symforce
try:
    symforce.set_epsilon_to_symbol()
except symforce.AlreadyUsedEpsilon:
    print("Already set symforce epsilon")
    pass 

import symforce.symbolic as sf

# ================================================= #
#                    PARAMETERS                     #
# ================================================= #

kitti_seq = '0028'
start_idx = 1550

N_SHIFT = 10
N_WINDOW = 100
GPS_RATE = 10

TRAJLEN = 2000   # 4544

MAX_BIAS = 50  # [m]
ATTACK_START = TRAJLEN // 2

ATTITUDE = False

# Lidar odometry covariance
ODOM_R_SIGMA = 0.01  # 
ODOM_T_SIGMA = 0.05  # [m]
#ODOM_SIGMA = np.array([0.05, 0.05, 0.05, 0.2, 0.2, 0.2])
ODOM_SIGMA = np.ones(6)
ODOM_SIGMA[:3] *= ODOM_R_SIGMA
ODOM_SIGMA[3:] *= ODOM_T_SIGMA

LIDAR_SIGMA = np.eye(6)
LIDAR_SIGMA[:3, :3] *= ODOM_R_SIGMA
LIDAR_SIGMA[3:, 3:] *= ODOM_T_SIGMA
lidar_sigmas = [sf.Matrix(LIDAR_SIGMA) for i in range(TRAJLEN)]

# ================================================= #
#                       SETUP                       #
# ================================================= #

# Load the data
print("Loading data...")
gtpath = os.path.join(os.getcwd(), '..', 'data', 'kitti', kitti_seq, 'oxts', 'data')
gt_enu, gt_Rs, gt_attitudes = process_kitti_gt(gtpath, start_idx=start_idx)

data_path = os.path.join(os.getcwd(), '..', 'data', 'kitti', kitti_seq, 'results', 'p2pl_icp')
lidar_Rs, lidar_ts, positions, lidar_covariances = load_icp_results(data_path, start_idx=start_idx)

# Satellite positions
# Uniformly arranged in a circle of radius R at constant altitude
R = 1e4
SAT_ALT = 1e4
N_SATS = 10
satpos_enu = np.zeros((N_SATS, 3))
satpos_enu[:,0] = np.cos(np.linspace(0, 2*np.pi, N_SATS, endpoint=False)) * R
satpos_enu[:,1] = np.sin(np.linspace(0, 2*np.pi, N_SATS, endpoint=False)) * R
satpos_enu[:,2] = SAT_ALT

# Compute threshold
alpha = 0.001  # False alarm (FA) rate
T = chi2.ppf(1-alpha, df=N_SATS*(N_WINDOW/GPS_RATE))

# Lidar odometry
lidar_odom = [None] * TRAJLEN
for i in range(TRAJLEN - 1):
    lidar_odom[i] = (lidar_Rs[i], lidar_ts[i])

# Spoofed trajectory
spoofed_pos = gt_enu[:TRAJLEN].copy()

gps_spoofing_biases = np.zeros(TRAJLEN)  
gps_spoofing_biases[ATTACK_START:] = np.linspace(0, MAX_BIAS, TRAJLEN-ATTACK_START)  # Ramping attack

spoofed_pos[:,0] += gps_spoofing_biases

# Spoofed ranges
PR_SIGMA = 6
spoofed_ranges = np.zeros((TRAJLEN, N_SATS))
for i in range(TRAJLEN):
    for j in range(N_SATS):
        spoofed_ranges[i,j] = np.linalg.norm(spoofed_pos[i] - satpos_enu[j]) + np.random.normal(0, PR_SIGMA)

N_RUNS = 100

# ================================================= #
#                     RUN FGO                       #
# ================================================= #

for i in range(N_RUNS):
    print("Run", i)
    # Regenerate ranges
    spoofed_ranges = np.zeros((TRAJLEN, N_SATS))
    for i in range(TRAJLEN):
        for j in range(N_SATS):
            spoofed_ranges[i,j] = np.linalg.norm(spoofed_pos[i] - satpos_enu[j]) + np.random.normal(0, PR_SIGMA)

    graph_positions = np.zeros((TRAJLEN, 3))
    init_poses = [sf.Pose3.identity()] * N_WINDOW
    init_poses[0] = sf.Pose3(sf.Rot3.from_rotation_matrix(gt_Rs[0]), sf.V3(gt_enu[0]))
    e_ranges = np.zeros((N_WINDOW//GPS_RATE, N_SATS))
    qs = np.zeros(TRAJLEN//GPS_RATE)

    for k in tqdm(range((TRAJLEN - N_WINDOW) // N_SHIFT)):
        #print(N_SHIFT * k, '/', TRAJLEN)
        window = slice(N_SHIFT * k, N_SHIFT * k + N_WINDOW)
        odom = lidar_odom[window]
        ranges = spoofed_ranges[window]
        #odom_sigmas = lidar_sigmas[window]

        result = fgo(init_poses, satpos_enu, ranges, odom, PR_SIGMA, ODOM_SIGMA, GPS_RATE, fix_first_pose=True, debug=False)

        # Extract optimized positions
        window_positions = np.zeros((N_WINDOW, 3))
        for i in range(N_WINDOW):
            pose = result.optimized_values["poses"][i]
            window_positions[i] = pose.position().flatten()   

        # Save trajectory
        graph_positions[N_SHIFT*k:N_SHIFT*(k+1)] = window_positions[:N_SHIFT]

        # Update initial poses
        init_poses[:-N_SHIFT] = result.optimized_values["poses"][N_SHIFT:]

        # Use lidar odometry to initialize latest new poses
        n_pose = result.optimized_values["poses"][-1]
        n_pose = sf.Pose3(sf.Rot3.from_rotation_matrix(n_pose.R.to_rotation_matrix()), sf.V3(n_pose.t))
        for i in range(N_SHIFT):
            lidar_T = sf.Pose3(sf.Rot3.from_rotation_matrix(lidar_Rs[N_SHIFT*k + N_WINDOW + i]), sf.V3(lidar_ts[N_SHIFT*k + N_WINDOW + i]))
            n_pose = lidar_T * n_pose
            init_poses[-N_SHIFT+i] = n_pose

        # Compute test statistic
        for i in range(N_WINDOW//GPS_RATE):
            for j in range(N_SATS):
                e_ranges[i,j] = np.linalg.norm(window_positions[::GPS_RATE][i] - satpos_enu[j])
        q = np.sum((ranges[::GPS_RATE] - e_ranges)**2) / PR_SIGMA**2
        #print("q = ", q)
        qs[k] = q

    OPT_TRAJLEN = N_SHIFT*(k+1)

    graph_positions = graph_positions[:OPT_TRAJLEN]
    gt_enu = gt_enu[:OPT_TRAJLEN]
    qs = qs[:OPT_TRAJLEN]

    # Compute statistics
    # RMSE error
    rmse_xyz = np.sqrt(np.mean((graph_positions - gt_enu)**2, axis=0))
    print("RMSE (xyz): ", rmse_xyz)
    print("RMSE (overall): ", rmse_xyz.mean())

    # Detection / time to detection
    print('Max test statistic: ', qs.max())
    attack_detected = np.any(qs > T)
    if attack_detected:
        t_detected = np.argmax(qs > T) * N_SHIFT
        print("Attack detected at t = ", t_detected)

    # Max errors
    max_xyz = np.max(np.abs(graph_positions - gt_enu), axis=0)
    print("Max error (xyz): ", max_xyz)

    # Mean errors


    # Plots
    # Trajectory
    
    # 