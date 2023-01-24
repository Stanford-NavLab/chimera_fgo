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

from chimera_fgo.symforce.residuals import position_residual, odometry_residual


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