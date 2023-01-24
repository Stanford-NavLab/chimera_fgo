"""Spoofing

Functions for different spoofing attacks

"""

import numpy as np




def global_ramp(nominal, max_bias, axis=0, start=None):
    """Global ramp attack

    Parameters
    ----------
    nominal : np.ndarray (N x 3)
        Nominal trajectory
    max_bias : float
        Maximum bias in meters
    axis : int, optional
        Axis to apply the bias to, by default 0
    start : int, optional
        Index to start the attack at, by default halfway through the trajectory

    Returns
    -------
    spoofed_pos : np.ndarray
        Spoofed trajectory

    """
    spoofed_pos = nominal.copy()
    traj_len = len(nominal)
    gps_spoofing_biases = np.zeros(traj_len)
    if start is None:
        start = traj_len // 2
    gps_spoofing_biases[start:] = np.linspace(0, max_bias, traj_len - start)  
    spoofed_pos[:,axis] += gps_spoofing_biases

    return spoofed_pos


def local_ramp(nominal, headings, max_bias, cross_track=True, start=None):
    """Local ramp attack

    Parameters
    ----------
    nominal : np.ndarray (N x 3)
        Nominal trajectory
    headings : np.array (N)
        Heading of the vehicle at each time step
    max_bias : float
        Maximum bias in meters
    cross_track : bool, optional
        Whether to apply the bias in the cross-track direction or along-track direction, by default True
    start : int, optional
        Index to start the attack at, by default halfway through the trajectory

    """
    spoofed_pos = nominal.copy()
    traj_len = len(nominal)
    gps_spoofing_biases = np.zeros(traj_len)
    if start is None:
        start = traj_len // 2
        
    gps_spoofing_biases[start:] = np.linspace(0, max_bias, traj_len - start)  
    spoofed_pos[:,0] += gps_spoofing_biases

    return spoofed_pos