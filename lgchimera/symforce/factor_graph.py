"""Factor graph

Symforce functions for factor graph optimization.

"""

import symforce

symforce.set_epsilon_to_symbol()

import numpy as np

import symforce.symbolic as sf
from symforce.values import Values
from symforce import typing as T
from symforce.opt.factor import Factor

# -----------------------------------------------------------------------------
# Residual functions
# -----------------------------------------------------------------------------
def range_residual(
    pose: sf.Pose3, satellite: sf.V3, range: sf.Scalar, epsilon: sf.Scalar
) -> sf.V1:
    """(Psuedo)range residual

    Parameters
    ----------
    pose : sf.Pose3
        Robot pose
    satellite : sf.V3
        Satellite position
    range : sf.Scalar
        Measured range
    epsilon : sf.Scalar
        Epsilon for numerical stability
    
    Returns
    -------
    sf.V1
        Residual

    """
    return sf.V1((pose.t - satellite).norm(epsilon=epsilon) - range)


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


# -----------------------------------------------------------------------------
# Build values
# -----------------------------------------------------------------------------
def build_values(num_poses, satpos, m_ranges, m_odometry):
    """
    """
    values = Values()

    # Poses
    values["poses"] = [sf.Pose3.identity()] * num_poses

    # Satellites
    values["satellites"] = [sf.V3(pos) for pos in satpos]

    # Ranges
    values["ranges"] = m_ranges


def factor_graph(sat_pos, ):
    """Factor graph with range and odometry factors

    Parameters
    ----------
    sat_pos : 

    ranges : 



    """
    values = Values()
    initial_values = Values(
        world_T_body=[sf.Pose3.identity()] * num_poses,
        world_t_landmark=[sf.V3(s_pos[0]), sf.V3(s_pos[1]), sf.V3(s_pos[2])],
        odometry=[sf.Pose3(R = sf.Rot3(), t = sf.V3(4, 3, 0)), 
                sf.Pose3(R = sf.Rot3(), t = sf.V3(4, -1, 0))],
        ranges=ranges.tolist(),
        sigmas=sf.V6(1, 1, 1, 1, 1, 1),
        epsilon=sf.numeric_epsilon,
    )

    factors = []

    # Range factors
    for i in range(num_poses):
        for j in range(num_satellites):
            factors.append(Factor(
                residual=range_residual,
                keys=[f"world_T_body[{i}]", f"world_t_landmark[{j}]", f"ranges[{i}][{j}]", "epsilon"],
            ))

    # Odometry factors
    for i in range(num_poses - 1):
        factors.append(Factor(
            residual=odometry_residual,
            keys=[f"world_T_body[{i}]", f"world_T_body[{i + 1}]", f"odometry[{i}]", "sigmas", "epsilon"],
        ))