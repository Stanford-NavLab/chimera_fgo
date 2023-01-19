"""Factor graph

Symforce functions for factor graph optimization.

"""

import symforce
try:
    symforce.set_epsilon_to_symbol()
except symforce.AlreadyUsedEpsilon:
    pass 

import symforce.symbolic as sf
from symforce.values import Values
from symforce import typing as T
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer


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
        Diagonal covariance of measurements
    
    Returns
    -------
    sf.V3
        Residual

    """
    return sf.M.diag(sigma.to_flat_list()).inv() * sf.V3(pose.t - m_pos)


def orientation_residual(
        pose: sf.Pose3, 
        m_ori: sf.Rot3, 
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


def build_factors(num_poses, num_satellites, gps_rate=1, fix_first_pose=False):
    """Build range and odometry factors

    """
    start_idx = gps_rate if fix_first_pose else 0
    for i in range(start_idx, num_poses, gps_rate):
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


def build_values_att(init_poses, satpos, m_ranges, m_odometry, m_attitudes, range_sigma, odom_sigma, att_sigma):
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
    values["attitudes"] = [sf.Rot3.from_rotation_matrix(R) for R in m_attitudes]

    values["range_sigma"] = range_sigma
    values["odom_sigma"] = odom_sigma
    values["att_sigma"] = att_sigma

    values["epsilon"] = sf.numeric_epsilon

    return values


def build_factors_att(num_poses, num_satellites, gps_rate=1, fix_first_pose=False):
    """Build range and odometry factors

    """
    start_idx = gps_rate if fix_first_pose else 0
    for i in range(start_idx, num_poses, gps_rate):
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
        yield Factor(
                residual=orientation_residual,
                keys=[
                    f"poses[{i}]",
                    f"attitudes[{i}]",
                    "att_sigma",
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


def build_values_eph(init_poses, satpos, m_ranges, m_odometry, range_sigma, odom_sigma):
    """Build values for factor graph

    Parameters
    ----------
    num_poses : int
        Number of poses 
    satpos : np.ndarray (num_poses x N_sats x 3)
        Satellite positions over time
    m_ranges : np.ndarray
        Measured ranges
    m_odometry : list of tuple (R, t)
        Measured odometry

    """
    values = Values()

    values["poses"] = init_poses
    values["satellites"] = satpos

    values["ranges"] = m_ranges.tolist()
    values["odometry"] = [sf.Pose3(R = sf.Rot3.from_rotation_matrix(R), t = sf.V3(t)) for R, t in m_odometry]

    values["range_sigma"] = range_sigma
    values["odom_sigma"] = odom_sigma

    values["epsilon"] = sf.numeric_epsilon

    return values


def build_factors_eph(num_poses, num_satellites, gps_rate=1, fix_first_pose=False):
    """Build range and odometry factors

    """
    start_idx = gps_rate if fix_first_pose else 0
    for i in range(start_idx, num_poses, gps_rate):
        for j in range(num_satellites):
            yield Factor(
                residual=range_residual,
                keys=[
                    f"poses[{i}]",
                    #"poses",
                    f"satellites[{i}][{j}]",
                    #"satellites",
                    f"ranges[{i}][{j}]",
                    #"ranges",
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


def fgo(init_pos, sat_pos, m_ranges, m_odom, range_sigma, odom_sigma, gps_rate, fix_first_pose=False, debug=False):
    """Factor graph optimization

    Parameters
    ----------

    """
    N_POSES = len(init_pos)
    N_SATS = len(sat_pos)

    # Build values and factors
    values = build_values(init_pos, sat_pos, m_ranges, m_odom, 
                        range_sigma, odom_sigma)
    factors = build_factors(N_POSES, N_SATS, gps_rate=gps_rate, fix_first_pose=fix_first_pose)

    # Select the keys to optimize - the rest will be held constant
    start_idx = 1 if fix_first_pose else 0
    optimized_keys = [f"poses[{i}]" for i in range(start_idx, N_POSES)]

    # Create the optimizer
    optimizer = Optimizer(
        factors=factors,
        optimized_keys=optimized_keys,
        # Return problem stats for every iteration
        debug_stats=debug,
        # Customize optimizer behavior
        params=Optimizer.Params(verbose=debug, initial_lambda=1e4, lambda_down_factor=0.5),
    )

    # Solve and return the result
    result = optimizer.optimize(values) 
    return result


def fgo_att(init_pos, sat_pos, m_ranges, m_odom, m_att, range_sigma, odom_sigma, att_sigma, gps_rate, fix_first_pose=False, debug=False):
    """Factor graph optimization

    Parameters
    ----------

    """
    print("attitude optimization")
    N_POSES = len(init_pos)
    N_SATS = len(sat_pos)

    # Build values and factors
    # values = build_values(init_pos, sat_pos, m_ranges, m_odom, 
    #                     range_sigma, odom_sigma)
    # factors = build_factors(N_POSES, N_SATS, gps_rate=gps_rate, fix_first_pose=fix_first_pose)
    values = build_values_att(init_pos, sat_pos, m_ranges, m_odom, m_att,
                        range_sigma, odom_sigma, att_sigma)
    factors = build_factors_att(N_POSES, N_SATS, gps_rate=gps_rate, fix_first_pose=fix_first_pose)

    # Select the keys to optimize - the rest will be held constant
    start_idx = 1 if fix_first_pose else 0
    optimized_keys = [f"poses[{i}]" for i in range(start_idx, N_POSES)]

    # Create the optimizer
    optimizer = Optimizer(
        factors=factors,
        optimized_keys=optimized_keys,
        # Return problem stats for every iteration
        debug_stats=debug,
        # Customize optimizer behavior
        params=Optimizer.Params(verbose=debug, initial_lambda=1e4, lambda_down_factor=0.5),
    )

    # Solve and return the result
    result = optimizer.optimize(values) 
    return result


def fgo_eph(init_pos, sat_pos, m_ranges, m_odom, range_sigma, odom_sigma, gps_rate, fix_first_pose=False, debug=False):
    """Factor graph optimization

    Parameters
    ----------

    """
    N_POSES = len(init_pos)
    N_SATS = len(sat_pos[0])

    # Build values and factors
    values = build_values_eph(init_pos, sat_pos, m_ranges, m_odom, 
                        range_sigma, odom_sigma)
    factors = build_factors_eph(N_POSES, N_SATS, gps_rate=gps_rate, fix_first_pose=fix_first_pose)

    # Select the keys to optimize - the rest will be held constant
    start_idx = 1 if fix_first_pose else 0
    optimized_keys = [f"poses[{i}]" for i in range(start_idx, N_POSES)]

    # Create the optimizer
    optimizer = Optimizer(
        factors=factors,
        optimized_keys=optimized_keys,
        # Return problem stats for every iteration
        debug_stats=debug,
        # Customize optimizer behavior
        params=Optimizer.Params(verbose=debug, initial_lambda=1e4, lambda_down_factor=0.5),
    )

    # Solve and return the result
    result = optimizer.optimize(values) 
    return result