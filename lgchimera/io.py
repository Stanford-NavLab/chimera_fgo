"""Utilities for I/O

"""

import numpy as np
import struct 
import os


def read_lidar_bin(binpath):
    """Read bin files containing LiDAR point cloud data 

    Parameters
    ----------
    binpath : str
        Path to folder containing bin files

    Returns
    -------
    PC_data : list of np.array (n_pts x 3)
        List of point clouds for each frame

    """
    PC_data = []
    size_float = 4
    for i in range(len(os.listdir(binpath))):
        list_pcd = []
        bf = binpath+"/{i}.bin".format(i=str(i).zfill(10))

        try: 
            with open(bf, "rb") as f: 
                byte = f.read(size_float*4)
                while byte:
                    x, y ,z, intensity = struct.unpack("ffff", byte)
                    list_pcd.append([x, y, z])
                    byte = f.read(size_float * 4)
            np_pcd = np.asarray(list_pcd)
        except FileNotFoundError:
            print(f"file {i} wasn't found")
        
        PC_data.append(np_pcd)
    
    return PC_data


def read_gt(path):
    """Read txt files containing ground-truth information

    Parameters
    ----------
    path : str
        Path to folder containing txt files

    Returns
    -------
    gt_arr : np.array
        Ground-truth data

    """
    gt_arr = []
    for i in range(len(os.listdir(path))):
        pf = path+"/{i}.txt".format(i=str(i).zfill(10))

        with open(pf) as f:
            lines = f.read().splitlines()
            data = [float(x) for x in lines[0].split(' ')]
            gt_arr.append(data)
    
    return np.array(gt_arr)