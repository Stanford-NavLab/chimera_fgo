import numpy as np
import time
import os
import tracemalloc
import gc

import symforce
try:
    symforce.set_epsilon_to_symbol()
except symforce.AlreadyUsedEpsilon:
    print("Already set symforce epsilon")
    pass 

import symforce.symbolic as sf

from chimera_fgo.chimera_fgo import chimera_fgo


# ================================================= #
#                 USER PARAMETERS                   #
# ================================================= #

# kitti_seq = '0034'  # ['0018', '0027', '0028', '0034']
# MAX_BIAS = 0  # [m]  # [0, 20, 50, 100]
params = {}
N_RUNS = 10
params['MITIGATION'] = True
params['ATTITUDE'] = False
params['DEBUG'] = False
params['CLEAR_MEM'] = False

# ================================================= #
#                    PARAMETERS                     #
# ================================================= #

params['N_SHIFT'] = 10
params['N_WINDOW'] = 100
params['GPS_RATE'] = 10

params['TRAJLEN'] = 2000   

params['ATTACK_START'] = params['TRAJLEN'] // 2
params['ALPHA'] = 0.001  # False alarm (FA) rate

params['PR_SIGMA'] = 6.0  # [m]

# Lidar odometry covariance
ODOM_R_SIGMA = 0.01  # so(3) tangent space units (~= rad for small values)
ODOM_T_SIGMA = 0.05  # [m]
ODOM_SIGMA = np.ones(6)
ODOM_SIGMA[:3] *= ODOM_R_SIGMA
ODOM_SIGMA[3:] *= ODOM_T_SIGMA
params['ODOM_SIGMA'] = ODOM_SIGMA

# Attitude measurement standard deviations
ATT_SIGMA_SCALE = 0.2
ATT_SIGMA = ATT_SIGMA_SCALE * np.ones(3)  # so(3) tangent space units (~= rad for small values)
params['ATT_SIGMA'] = ATT_SIGMA

MAX_BIAS = 100
N_WINDOW = 100
MITIGATION = True
ATTACK_START = 1000

params['REPROCESS_WINDOW'] = 300

# ================================================= #
#                       SETUP                       #
# ================================================= #

# Start memory tracking
if params['DEBUG']: tracemalloc.start()

for kitti_seq in ['0027']:
    for MAX_BIAS in [200]:
        #for ATT_SIGMA_SCALE in [0.05, 0.1, 0.15, 0.2, 0.25]:
        for N_WINDOW in [20, 50, 100, 200, 300]:
        #for MITIGATION in [False]:

            params['kitti_seq'] = kitti_seq
            params['MAX_BIAS'] = MAX_BIAS
            params['N_WINDOW'] = N_WINDOW
            params['MITIGATION'] = MITIGATION
            #params['ATTACK_START'] = ATTACK_START
            RP_WINDOW = params['REPROCESS_WINDOW']

            print(f"Running {kitti_seq} with {MAX_BIAS}m bias: \
                  \n    mitigation = {params['MITIGATION']}, \
                  \n    attitude = {params['ATTITUDE']}, \
                  \n    window size = {params['N_WINDOW']}, \
                  \n    reprocess window = {params['REPROCESS_WINDOW']}, \
                  \n    attack start = {params['ATTACK_START']}, \
                  \n    runs = {N_RUNS}")

            datestr = time.strftime("%Y-%m-%d")
            date_path = os.path.join(os.getcwd(), '..', 'results', kitti_seq, datestr)
            if not os.path.exists(date_path):
                os.makedirs(date_path)

            fname = f'fgo_{MAX_BIAS}m_{N_RUNS}runs_{N_WINDOW}w_{RP_WINDOW}rp'
            if not params['MITIGATION']: fname += '_blind' 
            if params['ATTITUDE']: fname = 'att_' + fname

            results_path = os.path.join(date_path, fname)
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            else:
                # Run with same name already exists, so add a number to the end
                i = 1
                while os.path.exists(results_path + f'_{i}'):
                    i += 1
                results_path += f'_{i}'
                os.makedirs(results_path)

            for run_i in range(N_RUNS):

                print("  Run number", run_i)
                output = chimera_fgo(params)

                if params['DEBUG']: 
                    current, peak = tracemalloc.get_traced_memory()
                    print(f"  Current memory usage is {current / 10**6:.2f} MB; Peak was {peak / 10**6:.2f} MB")
            
                # Save the output to file
                timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
                np.savez_compressed(os.path.join(results_path, f'run_{timestr}.npz'), **output, **params)

                # Clear the memory
                if params['CLEAR_MEM']:
                    del output
                    gc.collect()

            # Track memory
            # tracemalloc.start()
            # current, peak = tracemalloc.get_traced_memory()
            # print(f"  Current memory usage is {current / 10**6:.2f} MB; Peak was {peak / 10**6:.2f} MB")
            # tracemalloc.stop()