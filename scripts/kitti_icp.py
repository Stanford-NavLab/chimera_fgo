import os

import lgchimera.general as general
from lgchimera.io import read_lidar_bin, read_gt

binpath = os.path.join(os.getcwd(), '..', 'data', 'kitti', '2011_09_30_drive_0028_sync', 'velodyne_points', 'data')
PC_data_all = read_lidar_bin(binpath)