import numpy as np
import time
import os
from scipy.stats import chi2
from tqdm import tqdm

from chimera_fgo.util.kitti import process_kitti_gt, load_icp_results, load_sv_positions
from chimera_fgo.symforce.factor_graph import fgo_eph

import symforce
try:
    symforce.set_epsilon_to_symbol()
except symforce.AlreadyUsedEpsilon:
    print("Already set symforce epsilon")
    pass 

import symforce.symbolic as sf


# ================================================= #
#                 USER PARAMETERS                   #
# ================================================= #

kitti_seq = '0027'  # ['0018', '0027', '0028', '0034']
MAX_BIAS = -50  # [m]

# ================================================= #
#                    PARAMETERS                     #
# ================================================= #

if kitti_seq == '0028':
    start_idx = 1550
else:
    start_idx = 0

N_SHIFT = 10
N_WINDOW = 100
GPS_RATE = 10

TRAJLEN = 2000   

ATTACK_START = TRAJLEN // 2
ALPHA = 0.001  # False alarm (FA) rate

PR_SIGMA = 6.0  # [m]
PR_SPOOFED_SIGMA = 1e5  # [m]

# Lidar odometry covariance
ODOM_R_SIGMA = 0.01  # so(3) tangent space units
ODOM_T_SIGMA = 0.05  # [m]
ODOM_SIGMA = np.ones(6)
ODOM_SIGMA[:3] *= ODOM_R_SIGMA
ODOM_SIGMA[3:] *= ODOM_T_SIGMA

# ================================================= #
#                       SETUP                       #
# ================================================= #

for i in range(100):
    if i < 50:
        MAX_BIAS = -50
    else:
        MAX_BIAS = 50

    #print("Running FGO on KITTI sequence", kitti_seq, "with max bias of", MAX_BIAS, "meters")
    print("Run number", i)

    # Load the data
    print("Loading data...")
    gtpath = os.path.join(os.getcwd(), '..', 'data', 'kitti', kitti_seq, 'oxts', 'data')
    gt_enu, gt_Rs, gt_attitudes = process_kitti_gt(gtpath, start_idx=start_idx)

    data_path = os.path.join(os.getcwd(), '..', 'data', 'kitti', kitti_seq, 'icp')
    lidar_Rs, lidar_ts, positions, lidar_covariances = load_icp_results(data_path, start_idx=start_idx)

    # Load satellite positions
    sv_path = os.path.join(os.getcwd(), '..', 'data', 'kitti', kitti_seq, 'svs')
    sv_positions = load_sv_positions(gtpath, sv_path, kitti_seq, start_idx=start_idx)
    N_SATS = len(sv_positions[0])

    # Compute threshold
    T = chi2.ppf(1-ALPHA, df=N_SATS*(N_WINDOW/GPS_RATE))

    # Lidar odometry
    lidar_odom = [None] * TRAJLEN
    for i in range(TRAJLEN - 1):
        lidar_odom[i] = (lidar_Rs[i], lidar_ts[i])

    # Spoofed trajectory
    spoofed_pos = gt_enu[:TRAJLEN].copy()
    gps_spoofing_biases = np.zeros(TRAJLEN)  
    gps_spoofing_biases[ATTACK_START:] = np.linspace(0, MAX_BIAS, TRAJLEN-ATTACK_START)  # Ramping attack
    spoofed_pos[:,0] -= gps_spoofing_biases

    # Spoofed ranges
    spoofed_ranges = np.zeros((TRAJLEN, N_SATS))
    for i in range(TRAJLEN):
        svs = sv_positions[i]
        for j in range(N_SATS):
            spoofed_ranges[i,j] = np.linalg.norm(spoofed_pos[i] - svs[j]) + np.random.normal(0, PR_SIGMA)


    # ================================================= #
    #                     RUN FGO                       #
    # ================================================= #

    graph_positions = np.zeros((TRAJLEN, 3))
    init_poses = [sf.Pose3.identity()] * N_WINDOW
    init_poses[0] = sf.Pose3(sf.Rot3.from_rotation_matrix(gt_Rs[0]), sf.V3(gt_enu[0]))
    e_ranges = np.zeros((N_WINDOW//GPS_RATE, N_SATS))
    qs = np.zeros(TRAJLEN//GPS_RATE)

    range_sigma = PR_SIGMA
    authentic = True

    for k in (pbar := tqdm(range((TRAJLEN - N_WINDOW) // N_SHIFT))):

        window = slice(N_SHIFT * k, N_SHIFT * k + N_WINDOW)
        odom = lidar_odom[window]
        ranges = spoofed_ranges[window]

        satpos_enu = np.array(sv_positions[window])
        satpos = [[sf.V3(satpos_enu[i][j]) for j in range(N_SATS)] for i in range(N_WINDOW)]

        result = fgo_eph(init_poses, satpos, ranges, odom, range_sigma, ODOM_SIGMA, GPS_RATE, fix_first_pose=True, debug=False)

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
            svs = sv_positions[N_SHIFT * k + i * GPS_RATE]
            for j in range(N_SATS):
                e_ranges[i,j] = np.linalg.norm(window_positions[::GPS_RATE][i] - svs[j])
        q = np.sum((ranges[::GPS_RATE] - e_ranges)**2) / range_sigma**2
        pbar.set_description(f"q = {q:.2f}")
        qs[k] = q

        # Mitigation
        if authentic:
            if q > T:
                #print("Attack detected at ", N_SHIFT * k)
                range_sigma = PR_SPOOFED_SIGMA
                authentic = False


    OPT_TRAJLEN = N_SHIFT*(k+1)


    # ================================================= #
    #                   SAVE RESULTS                    #
    # ================================================= #

    graph_positions_plot = graph_positions[:OPT_TRAJLEN]
    spoofed_pos_plot = spoofed_pos[:OPT_TRAJLEN]
    qs_plot = qs[:OPT_TRAJLEN//N_SHIFT]

    spoofed = 'nominal' if MAX_BIAS == 0 else 'spoofed'
    results_path = os.path.join(os.getcwd(), '..', 'data', 'kitti', kitti_seq, 'results', spoofed)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    timestr = time.strftime("%Y-%m-%d_%H%M")

    if MAX_BIAS == 0:
        np.savez_compressed(os.path.join(results_path, timestr+'_fgo.npz'), positions=graph_positions_plot, qs=qs_plot)
    else:
        np.savez_compressed(os.path.join(results_path, timestr+f'_fgo_{MAX_BIAS}m.npz'), positions=graph_positions_plot, spoofed=spoofed_pos_plot, qs=qs_plot)