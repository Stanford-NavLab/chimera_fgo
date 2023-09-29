Spoofing-resilient LiDAR-GPS Factor Graph Optimization (FGO) with Chimera authentication.

Paper: https://ieeexplore.ieee.org/abstract/document/10139945
```
@inproceedings{dai2023spoofing,
  title={Spoofing-Resilient LiDAR-GPS Factor Graph Localization with Chimera Authentication},
  author={Dai, Adam and Mina, Tara and Kanhere, Ashwin and Gao, Grace},
  booktitle={2023 IEEE/ION Position, Location and Navigation Symposium (PLANS)},
  pages={470--480},
  year={2023},
  organization={IEEE}
}
```

## Setup

Clone the GitHub repository:

    git clone https://github.com/Stanford-NavLab/chimera_fgo.git

Create and activate conda environment:

    conda create -n chimera python=3.9
    conda activate chimera
    
Install dependencies:

    pip install -r requirements.txt
   
Install `chimera` locally from directory containing `setup.py`
   
    pip install -e .

## Data
We use data from the KITTI dataset for our experiments. The pre-processed data can be found at this Google drive [link](https://drive.google.com/drive/folders/17K4qRPXs8pU1r1awYJjoh5m6WK9RJrGX?usp=sharing). 
Inside `chimera_fgo`, create a `data` folder and with the following structure (you may have to unzip the `oxts` folders):
```
chimera_fgo
|  data
   |  kitti
      |  0018
         |  icp
         |  svs
         |  oxts      
      |  0027
      ...
```

## Running experiments
The `ephermeris_fgo.ipynb` notebook steps through an example of Chimera FGO.

The script `testing_script.py` is used to run Monte Carlo simulations over various different FGO settings (window size, spoofing attack size), and calls the `chimera_fgo` function in `chimera_fgo.py`.
