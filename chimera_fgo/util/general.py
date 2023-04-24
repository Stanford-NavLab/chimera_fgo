"""General utilities

"""

import os, sys
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


class SuppressPrint:
    """Utility for suppressing print statements
    
    Ex.)
        with SuppressPrint():
            foo()
    
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def NED_to_ENU(P):
    """Convert a set of 3D points from NED to ENU

    Parameters
    ----------
    P : np.array (n_pts x 3)
        NED points to convert

    Returns
    -------
    np.array (n_pts x 3)
        points in ENU

    """
    P[:,[0,1]] = P[:,[1,0]]  # Swap x and y
    P[:,2] = -P[:,2]  # Negate z
    return P


def lla_to_ecef(lat, lon, alt):
    """LLA to ECEF conversion.

    Parameters
    ----------
    lat : float
        Latitude in degrees (N°).
    lat : float
        Longitude in degrees (E°).
    alt : float
        Altitude in meters.

    Returns
    -------
    ecef : np.ndarray
        ECEF coordinates corresponding to input LLA.

    Notes
    -----
    Based on code from https://github.com/Stanford-NavLab/gnss_lib_py.

    """
    A = 6378137  # Semi-major axis (radius) of the Earth [m].
    E1SQ = 6.69437999014 * 0.001  # First esscentricity squared of Earth (not orbit).
    lat = np.deg2rad(lat); lon = np.deg2rad(lon)
    xi = np.sqrt(1 - E1SQ * np.sin(lat)**2)
    x = (A / xi + alt) * np.cos(lat) * np.cos(lon)
    y = (A / xi + alt) * np.cos(lat) * np.sin(lon)
    z = (A / xi * (1 - E1SQ) + alt) * np.sin(lat)
    ecef = np.array([x, y, z]).T
    return ecef


def ecef2enu(x, y, z, lat_ref, lon_ref, alt_ref):
    """ECEF to ENU
    
    Convert ECEF (m) coordinates to ENU (m) about reference latitude (N°) and longitude (E°).

    Parameters
    ----------
    x : float
        ECEF x-coordinate
    y : float
        ECEF y-coordinate
    z : float
        ECEF z-coordinate
    lat_ref : float
        Reference latitude (N°) 
    lon_ref : float
        Reference longitude (E°) 
    alt_ref : float
        Reference altitude (m)
    
    Returns
    -------
    x : float
        ENU x-coordinate
    y : float
        ENU y-coordinate
    z : float
        ENU z-coordinate

    """
    ecef_ref = lla_to_ecef(lat_ref, lon_ref, alt_ref)
    lat_ref = np.deg2rad(lat_ref)
    lon_ref = np.deg2rad(lon_ref + 360)
    C = np.zeros((3,3))
    C[0,0] = -np.sin(lat_ref)*np.cos(lon_ref)
    C[0,1] = -np.sin(lat_ref)*np.sin(lon_ref)
    C[0,2] = np.cos(lat_ref)

    C[1,0] = -np.sin(lon_ref)
    C[1,1] = np.cos(lon_ref)
    C[1,2] = 0

    C[2,0] = np.cos(lat_ref)*np.cos(lon_ref)
    C[2,1] = np.cos(lat_ref)*np.sin(lon_ref)
    C[2,2] = np.sin(lat_ref)

    x, y, z = np.dot(C, np.array([x, y, z]) - ecef_ref)

    return x, y, z


def normalize(v):
    """Normalize numpy vector

    Parameters
    ----------
    v : np.array 
        Vector to normalize

    Returns
    -------
    np.array 
        Normalized vector

    """
    return v / np.linalg.norm(v)


def pc_plot_trace(P, color=None, size=2, opacity=1.0):
    """Generate plotly plot trace for point cloud

    Parameters
    ----------
    P : np.array (N x 3)
        Point cloud

    Returns
    -------
    go.Scatter3d
        Scatter plot trace

    """
    if color is None:
        color = P[:,2]
    return go.Scatter3d(x=P[:,0], y=P[:,1], z=P[:,2], 
        mode='markers', opacity=opacity, marker=dict(size=size, color=color))


def pose_plot_trace(R, t):
    """Generate plotly plot trace for a 3D pose (position and orientation)

    Parameters
    ----------
    R : np.array (3 x 3)
        Rotation matrix representing orientation
    t : np.array (3)
        Translation vector representing position
    
    Returns
    -------
    list
        List containing traces for plotting 3D pose

    """
    point = go.Scatter3d(x=[t[0]], y=[t[1]], z=[t[2]], 
                mode='markers', marker=dict(size=5))
    xs = []; ys = []; zs = []
    for i in range(3):
        xs += [t[0], t[0] + R[0,i], None]
        ys += [t[1], t[1] + R[1,i], None]
        zs += [t[2], t[2] + R[2,i], None]
    lines = go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", showlegend=False)
    return [point, lines]


def trajectory_plot_trace(Rs, ts, color="red", scale=1.0):
    """Generate plotly plot trace for a 3D trajectory of poses

    Parameters
    ----------
    Rs : np.array (3 x 3 x N)
        Sequence of orientations
    ts : np.array (N x 3)
        Sequence of positions
    
    Returns
    -------
    list
        List containing traces for plotting 3D trajectory
    
    """
    points = go.Scatter3d(x=[ts[:,0]], y=[ts[:,1]], z=[ts[:,2]], showlegend=False)#, mode='markers', marker=dict(size=5))
    xs = []; ys = []; zs = []
    for i in range(len(ts)):
        for j in range(3):
            xs += [ts[i,0], ts[i,0] + scale*Rs[0,j,i], None]
            ys += [ts[i,1], ts[i,1] + scale*Rs[1,j,i], None]
            zs += [ts[i,2], ts[i,2] + scale*Rs[2,j,i], None]
    lines = go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color), showlegend=False)
    return [points, lines]


