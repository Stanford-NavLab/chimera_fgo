"""FGO

"""

import numpy as np
import time
import os
from scipy.stats import chi2
from tqdm import tqdm
import tracemalloc
import gc

import symforce
try:
    symforce.set_epsilon_to_symbol()
except symforce.AlreadyUsedEpsilon:
    print("Already set symforce epsilon")
    pass 

import symforce.symbolic as sf

from chimera_fgo.util.kitti import process_kitti_gt, load_icp_results, load_sv_positions
from chimera_fgo.util.geometry import so3_expmap
from chimera_fgo.odometry import odometry
from chimera_fgo.symforce.tc_fgo import tc_fgo
from chimera_fgo.symforce.att_fgo import att_fgo
from chimera_fgo.symforce.fgo_lidar_only import fgo_lidar_only


def chimera_fgo(params):
    """
    
    """
    # ================================================= #
    #                 UNPACK PARAMETERS                 #
    # ================================================= #
    kitti_seq = params['kitti_seq']
    MAX_BIAS = params['MAX_BIAS']

    start_idx = 1550 if kitti_seq == '0028' else 0

    N_SHIFT = params['N_SHIFT']
    N_WINDOW = params['N_WINDOW']
    GPS_RATE = params['GPS_RATE']

    TRAJLEN = params['TRAJLEN']

    MITIGATION = params['MITIGATION']
    ATTACK_START = params['ATTACK_START']
    ALPHA = params['ALPHA']

    PR_SIGMA = params['PR_SIGMA']

    # Lidar odometry covariance
    ODOM_SIGMA = params['ODOM_SIGMA']

    ATTITUDE = params['ATTITUDE']

    # Attitude measurement standard deviations
    ATT_SIGMA = params['ATT_SIGMA']

    DEBUG = params['DEBUG']

    # ================================================= #
    #                     SETUP                         #
    # ================================================= #

    # Load the data
    print("Loading data...")
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'kitti', kitti_seq)
    gtpath = os.path.join(data_path, 'oxts', 'data')
    gt_enu, gt_Rs, gt_attitudes = process_kitti_gt(gtpath, start_idx=start_idx)

    icp_path = os.path.join(data_path, 'icp')
    lidar_Rs, lidar_ts, lidar_positions, lidar_covariances = load_icp_results(icp_path, start_idx=start_idx)

    # Load satellite positions
    sv_path = os.path.join(data_path, 'svs')
    sv_positions = load_sv_positions(gtpath, sv_path, kitti_seq, start_idx=start_idx)
    N_SATS = len(sv_positions[0])

    # Compute threshold
    T = chi2.ppf(1-ALPHA, df=N_SATS*(N_WINDOW/GPS_RATE))

    # Lidar odometry
    lidar_odom = [None] * TRAJLEN
    for i in range(TRAJLEN):
        lidar_odom[i] = (lidar_Rs[i], lidar_ts[i])

    # Spoofed trajectory
    spoofed_pos = gt_enu[:TRAJLEN].copy()
    gps_spoofing_biases = np.zeros(TRAJLEN)  
    gps_spoofing_biases[ATTACK_START:] = np.linspace(0, MAX_BIAS, TRAJLEN-ATTACK_START)  # Ramping attack
    spoofed_pos[:,0] += gps_spoofing_biases

    # ================================================= #
    #            GENERATE RANGES AND ATTITUDES          #
    # ================================================= #

    # Noisy attitudes
    m_rotations = [None] * TRAJLEN
    for i in range(TRAJLEN):
        noise = np.random.normal(0, ATT_SIGMA, size=3)
        m_rotations[i] = so3_expmap(noise) @ gt_Rs[i] 

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
    graph_attitudes = np.zeros((TRAJLEN, 3))
    init_poses = [sf.Pose3.identity()] * N_WINDOW
    init_poses[0] = sf.Pose3(sf.Rot3.from_rotation_matrix(gt_Rs[0]), sf.V3(gt_enu[0]))
    e_ranges = np.zeros((N_WINDOW//GPS_RATE, N_SATS))
    qs = np.zeros(TRAJLEN//GPS_RATE)

    authentic = True
    iter_times = []
    k_max = (TRAJLEN - N_WINDOW) // N_SHIFT + 1

    # Start memory tracing
    if params['DEBUG'] and not tracemalloc.is_tracing(): tracemalloc.start()

    for k in (pbar := tqdm(range(k_max))):

        start_time = time.time()

        # Slice of current window
        window = slice(N_SHIFT * k, N_SHIFT * k + N_WINDOW)
        odom = lidar_odom[window][:-1]

        # Optimize
        ranges = spoofed_ranges[window]
        satpos_enu = np.array(sv_positions[window])
        satpos = [[sf.V3(satpos_enu[i][j]) for j in range(N_SATS)] for i in range(N_WINDOW)]
        if ATTITUDE:
            attitudes = m_rotations[window]
            result = att_fgo(init_poses, satpos, ranges, odom, attitudes, PR_SIGMA, ODOM_SIGMA, ATT_SIGMA, GPS_RATE, fix_first_pose=True, debug=False)
        else:
            result = tc_fgo(init_poses, satpos, ranges, odom, PR_SIGMA, ODOM_SIGMA, GPS_RATE, fix_first_pose=True, debug=False)

        # Extract optimized positions and orientations
        window_positions = np.zeros((N_WINDOW, 3))
        window_attitudes = np.zeros((N_WINDOW, 3))
        for i in range(N_WINDOW):
            pose = result.optimized_values["poses"][i]
            window_positions[i] = pose.position().flatten()   
            window_attitudes[i] = pose.rotation().to_yaw_pitch_roll().flatten()   

        # Compute test statistic
        for i in range(N_WINDOW//GPS_RATE):
            svs = sv_positions[N_SHIFT * k + i * GPS_RATE]
            for j in range(N_SATS):
                e_ranges[i,j] = np.linalg.norm(window_positions[::GPS_RATE][i] - svs[j])
        q = np.sum((ranges[::GPS_RATE] - e_ranges)**2) / PR_SIGMA**2
        pbar.set_description(f"q = {q:.2f}")
        qs[k] = q

        # Check spoofing
        if MITIGATION and q > T:
            print("Attack detected at ", N_SHIFT * k)
            authentic = False

            # Process the remainder of the trajectory with odometry only
            init_pose = (init_poses[0].R.to_rotation_matrix(), init_poses[0].t.flatten())
            odom = lidar_odom[N_SHIFT * k:-1]
            positions, rotations = odometry(init_pose, odom)
            graph_positions[N_SHIFT * k:] = positions
            break  

        # Save trajectory
        graph_positions[N_SHIFT*k:N_SHIFT*(k+1)] = window_positions[:N_SHIFT]
        graph_attitudes[N_SHIFT*k:N_SHIFT*(k+1)] = window_attitudes[:N_SHIFT]

        # Update initial poses
        init_poses[:-N_SHIFT] = result.optimized_values["poses"][N_SHIFT:]

        # Use lidar odometry to initialize latest new poses
        n_pose = result.optimized_values["poses"][-1]
        n_pose = sf.Pose3(sf.Rot3.from_rotation_matrix(n_pose.R.to_rotation_matrix()), sf.V3(n_pose.t))
        for i in range(N_SHIFT):
            lidar_T = sf.Pose3(sf.Rot3.from_rotation_matrix(lidar_Rs[N_SHIFT*k + N_WINDOW + i]), sf.V3(lidar_ts[N_SHIFT*k + N_WINDOW + i]))
            n_pose = n_pose * lidar_T
            init_poses[-N_SHIFT+i] = n_pose
        
        iter_times.append(time.time() - start_time) 
        
        # Clean up memory
        del result
        del odom
        del ranges
        del satpos
        if k != k_max - 1:
            del window_positions
            del window_attitudes
        gc.collect()

        if params['DEBUG']:
            # Print memory usage
            current, peak = tracemalloc.get_traced_memory()
            print(f"        Current memory usage is {current / 10**6:.2f} MB; Peak was {peak / 10**6:.2f} MB")

    if authentic:
        # Save remaining poses in current window
        graph_positions[N_SHIFT*(k+1):] = window_positions[N_SHIFT:]
        graph_attitudes[N_SHIFT*(k+1):] = window_attitudes[N_SHIFT:]

    avg_iter_time = np.mean(iter_times)
    graph_attitudes = graph_attitudes[:,[2,1,0]]

    # ================================================= #
    #                   SAVE RESULTS                    #
    # ================================================= #
    
    position_errors = graph_positions - gt_enu[:TRAJLEN]
    l2norm_errors = np.linalg.norm(position_errors, axis=1)

    output = {}
    output['gt_positions'] = gt_enu[:TRAJLEN]
    output['fgo_positions'] = graph_positions
    output['fgo_attitudes'] = graph_attitudes
    output['spoofed_positions'] = spoofed_pos
    output['qs'] = qs[:k_max]
    output['position_errors'] = position_errors
    output['l2norm_errors'] = l2norm_errors
    output['avg_iter_time'] = avg_iter_time
    output['threshold'] = T

    return output