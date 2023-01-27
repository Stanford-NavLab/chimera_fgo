"""Odometry

"""

import numpy as np


def odometry(init_pose, m_odom):
    """Odometry
    
    Process remainder of trajectory using odometry only

    Parameters
    ----------
    init_pose : tuple (R, t)
        Initial pose
    m_odom : list of tuple (R, t)
        Measured odometry

    Returns
    -------
    positions : np.ndarray (N x 3)
        Estimated positions
    rotation : list of np.ndarray (3 x 3)
        Estimated rotations
    
    """
    R_abs, t_abs = init_pose
    positions = np.zeros((len(m_odom) + 1, 3))
    positions[0] = t_abs
    rotations = [R_abs]

    for i, (R, t) in enumerate(m_odom):
        t_abs += R_abs @ t
        R_abs = R @ R_abs
        positions[i+1] = t_abs
        rotations.append(R_abs)

    return positions, rotations