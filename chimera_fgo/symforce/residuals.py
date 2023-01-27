"""Residuals

Residual functions for SymForce factor graphs

"""

import symforce
try:
    symforce.set_epsilon_to_symbol()
except symforce.AlreadyUsedEpsilon:
    pass 

import symforce.symbolic as sf
from symforce import typing as T


def range_residual(
        pose: sf.Pose3, 
        satellite: sf.V3, 
        range: sf.Scalar, 
        sigma: sf.Scalar, 
        epsilon: sf.Scalar) -> sf.V1:
    """(Psuedo)range residual

    Parameters
    ----------
    pose : sf.Pose3
        Robot pose
    satellite : sf.V3
        Satellite position
    range : sf.Scalar
        Measured range
    sigma : sf.Scalar
        Standard deviation of range measurement
    epsilon : sf.Scalar
        Epsilon for numerical stability
    
    Returns
    -------
    sf.V1
        Residual

    """
    return sf.V1((pose.t - satellite).norm(epsilon=epsilon) - range) / sigma


def position_residual(
        pose: sf.Pose3, 
        m_pos: sf.V3, 
        sigma: sf.V3) -> sf.V3:
    """Position measurement residual

    Parameters
    ----------
    pose : sf.Pose3
        Robot pose
    m_pos : sf.V3
        Position measurement
    sigma : sf.V3
        Diagonal measurement standard deviation
    
    Returns
    -------
    sf.V3
        ResidualS

    """
    return sf.M.diag(sigma.to_flat_list()).inv() * sf.V3(pose.t - m_pos)


def orientation_residual(
        pose: sf.Pose3, 
        m_ori: sf.Rot3, 
        sigma: sf.V3, 
        epsilon: sf.Scalar) -> sf.V3:
    """Orientation measurement residual

    Parameters
    ----------
    pose : sf.Pose3
        Robot pose
    m_ori : sf.Rot3
        Measured orientation as a rotation matrix
    sigma : sf.V3
        Diagonal measurement standard deviation
    
    Returns
    -------
    sf.V3
        Residual

    """
    tangent_error = pose.R.local_coordinates(m_ori, epsilon=epsilon)
    return sf.M.diag(sigma.to_flat_list()).inv() * sf.V3(tangent_error) 


def odometry_residual(
    world_T_a: sf.Pose3,
    world_T_b: sf.Pose3,
    a_T_b: sf.Pose3,
    diagonal_sigmas: sf.V6,
    epsilon: sf.Scalar,
) -> sf.V6:
    """
    Residual on the relative pose between two timesteps of the robot.
    Args:
        world_T_a: First pose in the world frame
        world_T_b: Second pose in the world frame
        a_T_b: Relative pose measurement between the poses
        diagonal_sigmas: Diagonal standard deviation of the tangent-space error
        epsilon: Small number for singularity handling
    """
    a_T_b_predicted = world_T_a.inverse() * world_T_b
    tangent_error = a_T_b_predicted.local_coordinates(a_T_b, epsilon=epsilon)
    return T.cast(sf.V6, sf.M.diag(diagonal_sigmas.to_flat_list()).inv() * sf.V6(tangent_error))


# def odometry_residual(
#     world_T_a: sf.Pose3,
#     world_T_b: sf.Pose3,
#     a_T_b: sf.Pose3,
#     sigma: sf.Matrix66,
#     epsilon: sf.Scalar,
# ) -> sf.V6:
#     """
#     Residual on the relative pose between two timesteps of the robot.
#     Args:
#         world_T_a: First pose in the world frame
#         world_T_b: Second pose in the world frame
#         a_T_b: Relative pose measurement between the poses
#         diagonal_sigmas: Diagonal standard deviation of the tangent-space error
#         epsilon: Small number for singularity handling
#     """
#     a_T_b_predicted = world_T_a.inverse() * world_T_b
#     tangent_error = a_T_b_predicted.local_coordinates(a_T_b, epsilon=epsilon)
#     return T.cast(sf.V6, sigma.inv() * sf.V6(tangent_error))