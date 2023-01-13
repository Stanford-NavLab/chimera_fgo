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
from symforce.opt.optimizer import Optimizer

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
def build_values(init_poses, satpos, m_ranges, m_odometry, range_sigma, odom_sigma):
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

    values["poses"] = init_poses
    values["satellites"] = [sf.V3(pos) for pos in satpos]

    values["ranges"] = m_ranges.tolist()
    values["odometry"] = [sf.Pose3(R = sf.Rot3.from_rotation_matrix(R), t = sf.V3(t)) for R, t in m_odometry]

    values["range_sigma"] = range_sigma
    values["odom_sigma"] = odom_sigma

    values["epsilon"] = sf.numeric_epsilon

    return values


def build_factors(num_poses, num_satellites, gps_rate=1):
    """Build range and odometry factors

    """
    for i in range(0, num_poses, gps_rate):
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


def fgo(init_pos, sat_pos, m_ranges, odom, odom_sigma, range_sigma, gps_rate, fix_first_pose=False):
    """Factor graph optimization

    Parameters
    ----------

    """
    N_POSES = len(init_pos)
    N_SATS = len(sat_pos)

    # Build values and factors
    values = build_values(init_pos, sat_pos, m_ranges, odom, 
                        range_sigma, odom_sigma)
    factors = build_factors(N_POSES, N_SATS, gps_rate=gps_rate)

    # Select the keys to optimize - the rest will be held constant
    if fix_first_pose:
        optimized_keys = [f"poses[{i}]" for i in range(1, N_POSES)]
    else:
        optimized_keys = [f"poses[{i}]" for i in range(N_POSES)]

    # Create the optimizer
    optimizer = Optimizer(
        factors=factors,
        optimized_keys=optimized_keys,
        # Return problem stats for every iteration
        debug_stats=False,
        # Customize optimizer behavior
        params=Optimizer.Params(verbose=False, initial_lambda=1e4, lambda_down_factor=0.5),
    )

    # Solve and return the result
    result = optimizer.optimize(values) 
    return result


