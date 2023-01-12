"""Factor graph

Symforce functions for factor graph optimization.

"""

import symforce
try:
    symforce.set_epsilon_to_symbol()
except symforce.AlreadyUsedEpsilon:
    pass 

import numpy as np

import symforce.symbolic as sf
from symforce.values import Values
from symforce import typing as T
from symforce.opt.factor import Factor

# -----------------------------------------------------------------------------
# Residual functions
# -----------------------------------------------------------------------------
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
        sigma: sf.V3, 
        epsilon: sf.Scalar) -> sf.V3:
    """Position measurement residual

    Parameters
    ----------
    pose : sf.Pose3
        Robot pose
    m_pos : sf.V3
        Position measurement
    sigma : sf.V3
        Diagonal covariance of measurements
    
    Returns
    -------
    sf.V3
        Residual

    """
    return sf.V3(pose.t - m_pos) / sigma


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
def build_values(num_poses, satpos, m_ranges, m_odometry, range_sigma, odom_sigma):
    """Build values for factor graph

    Parameters
    ----------
    num_poses : int
        Number of poses 
    satpos : np.ndarray
        Satellite positions
    m_ranges : np.ndarray
        Measured ranges
    m_odometry : list of tuple (R, t)
        Measured odometry

    """
    values = Values()

    values["poses"] = [sf.Pose3.identity()] * num_poses
    values["satellites"] = [sf.V3(pos) for pos in satpos]

    values["ranges"] = m_ranges.tolist()
    values["odometry"] = [sf.Pose3(R = sf.Rot3.from_rotation_matrix(R), t = sf.V3(t)) for R, t in m_odometry]

    values["range_sigma"] = range_sigma
    values["odom_sigma"] = odom_sigma

    values["epsilon"] = sf.numeric_epsilon

    return values


def build_factors(num_poses, num_satellites):
    """Build range and odometry factors

    """
    for i in range(num_poses):
        for j in range(num_satellites):
            yield Factor(
                residual=range_residual,
                keys=[
                    f"poses[{i}]",
                    f"satellites[{j}]",
                    f"ranges[{i}][{j}]",
                    "range_sigma",
                    "epsilon",
                ],
            )

    for i in range(num_poses - 1):
        yield Factor(
            residual=odometry_residual,
            keys=[
                f"poses[{i}]",
                f"poses[{i + 1}]",
                f"odometry[{i}]",
                "odom_sigma",
                "epsilon",
            ],
        )


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