#!/usr/bin/env python

"""Package setup file.
"""

from setuptools import setup, find_packages

setup(
    name='lgchimera',
    version='1.0',
    description='LiDAR GPS Fusion with Chimera Authentication',
    author='Adam Dai',
    author_email='adamdai97@gmail.com',
    url='https://github.com/adamdai/lidar-gps-chimera',
    packages=find_packages(),
)