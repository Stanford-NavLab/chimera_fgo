# lidar-gps-chimera

LiDAR GPS fusion with Chimera authentication

## Setup

Clone the GitHub repository:

    git clone https://github.com/adamdai/lidar-gps-chimera.git

Create conda environment:

    conda create -n chimera python=3.9

Active the environment:
   
    conda activate chimera
    
Install dependencies:

    pip install numpy scipy ipykernel ipympl plotly pandas open3d graphslam
   
Install `chimera` locally from directory containing `setup.py`
   
    pip install -e .
