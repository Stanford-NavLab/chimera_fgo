"""Registration

Functions and utilities for point-cloud registration.

"""

import numpy as np
import open3d as o3d
import time
from numba import njit

from chimera_fgo.geom_util import se3_expmap, se3_logmap


def initialize_source_and_target(source_data, target_data):
    """Initialize source and target
    
    Parameters
    ----------
    source_data : np.array (N x 3)
        Source point cloud array
    target_data : np.array (N x 3)
        Target point cloud array
    
    Returns
    -------
    source : PointCloud
        Source as Open3D point cloud
    target : PointCloud
        Target as Open3D point cloud

    """
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_data)
    source.estimate_normals()
    source.orient_normals_towards_camera_location()

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_data)
    target.estimate_normals()
    target.orient_normals_towards_camera_location()

    return source, target


def p2p_ICP(source, target, threshold, trans_init):
    """Point-to-Point ICP
    
    Parameters
    ----------
    
    
    """
    start_time = time.time()
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    eval_time = time.time() - start_time
    return reg_p2p, eval_time


def p2pl_ICP(source, target, threshold, trans_init):
    """Point-to-Plane ICP
    
    Parameters
    ----------
    
    """
    start_time = time.time()
    reg_p2pl = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    eval_time = time.time() - start_time
    return reg_p2pl, eval_time


@njit
def p2pl_ICP_with_covariance(source, target, threshold, trans_init, sigma=0.02, bias=0.05, Q_ini=0.1*np.eye(6)):
    """Point-to-Plane ICP with covariance
    
    Parameters
    ----------

    Returns
    -------
    reg_p2pl : RegistrationResult
        Registration result from Open3D
    Q_sensor : np.array (6 x 6)
        Sensor uncertainty covariance (first 3 rotation, last 3 translation)
    
    """
    # ICP
    # TODO: manually apply T_ini as in Brossard paper
    reg_p2pl = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    # Sensor uncertainty term
    T_rel = reg_p2pl.transformation 
    corrs = np.asarray(reg_p2pl.correspondence_set)
    target_normals = np.asarray(target.normals)
    source_points = np.asarray(source.points)

    B = np.zeros(6)
    A = np.zeros((6,6))
    for i,j in corrs:
        n = target_normals[j]
        p_T = (T_rel @ np.hstack((source_points[i], 1)))[:3]
        B_i = -np.hstack((np.cross(p_T, n), n))
        B += B_i
        A += B_i[:,None] @ B_i[:,None].T
    
    A_inv = np.linalg.inv(A)
    Q_sensor = sigma**2 * A_inv + A_inv @ B[:,None] * bias * B[:,None].T @ A_inv 

    # Initialization noise term
    T_ini = trans_init
    T_icp = T_rel

    # Set sigma points
    v_ini = 12 * [None]
    v_ini[:6] = np.hsplit(np.sqrt(6 * Q_ini), 6)
    v_ini[6:] = np.hsplit(-np.sqrt(6 * Q_ini), 6)

    # Propagate sigma points through
    T_ini_samples = 12 * [None]
    T_icp_samples = 12 * [None]
    v_icp = 12 * [None]

    for i, v in enumerate(v_ini):
        T_ini_samples[i] = T_ini @ se3_expmap(v.flatten())
        T_icp_samples[i] = T_ini_samples[i] @ p2pl_ICP(source, target, threshold=1, trans_init=T_ini_samples[i])[0].transformation
        v_icp[i] = se3_logmap(np.linalg.inv(T_icp) @ T_icp_samples[i])

    v_icp = np.array(v_icp)

    # Compute covariance 
    Q_init = (v_icp.T @ v_icp) / 12
    
    return reg_p2pl, Q_sensor + Q_init