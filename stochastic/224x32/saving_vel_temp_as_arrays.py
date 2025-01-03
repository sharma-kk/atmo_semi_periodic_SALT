import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from firedrake import *
import math
import time

print("loading coarse grained velocity and temp......",
    "current time:",time.strftime("%H:%M:%S", time.localtime()))

with CheckpointFile("../../deterministic/224x32/h5_files/coarse_grained_vel_temp_at_t=27.0_mesh_32_c_1_by_32.h5", 'r') as afile:
    mesh = afile.load_mesh("coarse_mesh")
    uc = afile.load_function(mesh, "coarse_vel") 
    thetac = afile.load_function(mesh, "coarse_temp")

print("finished loading! calculating vorticity from velocity",
    time.strftime("%H:%M:%S", time.localtime()))

data_file ='./vel_temp_data/u_theta_coarse_grained_arrrays.npz'
np.savez(data_file, vel_array = uc.dat.data, temp_array = thetac.dat.data)
