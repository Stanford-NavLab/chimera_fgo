"""Utilities for KITTI data

"""

import numpy as np
import pandas as pd
import os

from chimera_fgo.io import read_gt
from chimera_fgo.geom_util import euler_to_R
from chimera_fgo.general import lla_to_ecef, ecef2enu


def process_kitti_gt(path, start_idx=0):
    """Process KITTI ground-truth


    """
    gt_data = read_gt(path)
    gt_data = gt_data[start_idx:]

    # Get ground truth positions
    lla = gt_data[:,:3]

    # Convert to ENU
    ref_lla = lla[0]
    ecef = lla_to_ecef(*lla[0])
    gt_enu = np.zeros((len(lla),3))

    for i in range(len(lla)):
        ecef = lla_to_ecef(*lla[i])
        gt_enu[i] = ecef2enu(ecef[0], ecef[1], ecef[2], ref_lla[0], ref_lla[1], ref_lla[2])
    gt_enu = gt_enu[:,[1,0,2]]

    # Get ground truth attitudes
    #  0 - roll: 0=level, positive=left side up, range -pi..pi
    #  1 - pitch: 0=level, positive=front down, range -pi/2..pi/2
    #  2 - yaw: 0=east, positive=counter clockwise, range -pi..pi
    gt_attitudes = gt_data[:,3:6]

    # Convert to rotation matrices
    gt_Rs = len(gt_attitudes) * [None]
    for i, angles in enumerate(gt_attitudes):
        gt_Rs[i] = euler_to_R(angles)

    return gt_enu, gt_Rs, gt_attitudes


def load_icp_results(data_path, start_idx=0, ds_rate=10, Q_ini=0.01):
    """Load ICP results
    """
    run_name = 'start_{}_ds_{}_Q_ini_{}'.format(start_idx, ds_rate, Q_ini)
    Rs = np.load(os.path.join(data_path, 'lidar_Rs_'+run_name+'.npy'))
    ts = np.load(os.path.join(data_path, 'lidar_ts_'+run_name+'.npy'))
    pos = np.load(os.path.join(data_path, 'positions_'+run_name+'.npy'))
    covs = np.load(os.path.join(data_path, 'covariances_'+run_name+'.npy'))

    return Rs, ts, pos, covs


def load_sv_positions(gt_path, sv_path, kitti_seq, start_idx=0):
    """Load satellite positions
    """
    # Get reference position
    gt_data = read_gt(gt_path)
    gt_data = gt_data[start_idx:]
    ref_lla = gt_data[0,:3]

    # Get satellite positions
    svfile = os.path.join(sv_path, f'{kitti_seq}_saved_sats.csv')
    df = pd.read_csv(svfile)
    sv_positions = []  # list of satellite position arrays (in ENU) for each timestep

    # Group measurements by epoch
    df_grouped = df.groupby('times')
    # Iterate over epochs
    for i, (_, group) in enumerate(df_grouped):
        if i >= start_idx:
            sat_xyz = np.array([group.x, group.y, group.z]).T
            # Convert to ENU
            for i, ecef in enumerate(sat_xyz):
                sat_xyz[i] = ecef2enu(ecef[0], ecef[1], ecef[2], ref_lla[0], ref_lla[1], ref_lla[2])
            sv_positions.append(sat_xyz[:,[1,0,2]])
    
    return sv_positions
        
