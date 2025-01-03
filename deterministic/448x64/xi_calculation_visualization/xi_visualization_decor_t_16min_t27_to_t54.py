import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from firedrake import *
import math
import time

Nx = 7*64
Ny = 64
c_mesh = PeriodicRectangleMesh(Nx, Ny, 7, 1, direction="x")
c_mesh.name = "coarse_mesh"

V1c = VectorFunctionSpace(c_mesh, "CG", 1)

# Xi1 = Function(V1c)
# Xi2 = Function(V1c)
xi = Function(V1c)

xi_data = np.load('./xi_vec_data/xi_matrix_combined_calc_94_eigvec_c_1_by_64_decor_t_16min_t=27_to_t=45.npz')
xi_mat = xi_data['xi_mat']

outfile  = File("./results/xi_vecs_decor_t_16min_t=27_to_t=54.pvd")

n_xi = [0, 17, 38, 93] # printing xi's which explains 50 % , 70 % and 90 % of the total variance.

## 18, 39, and 94 EOFs explain 50, 70 and 90 % of the total variance

for i in n_xi:

    print(f'saving xi {i+1}')
    print("Local time:",time.strftime("%H:%M:%S", time.localtime()))
    
    xi.assign(0)
    xi.dat.data[:] = xi_mat[i,:]

    xi.rename("xi")

    outfile.write(xi)

