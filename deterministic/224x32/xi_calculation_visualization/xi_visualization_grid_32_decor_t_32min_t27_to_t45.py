import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from firedrake import *
import math
import time

Nx = 7*32
Ny = 32
c_mesh = PeriodicRectangleMesh(Nx, Ny, 7, 1, direction="x")
c_mesh.name = "coarse_mesh"

V1c = VectorFunctionSpace(c_mesh, "CG", 1)

# Xi1 = Function(V1c)
# Xi2 = Function(V1c)
xi = Function(V1c)

xi_data = np.load('./xi_vec_data/xi_matrix_combined_calc_51_eigvec_grid_32_c_1_by_32_decor_t_32min_t=27_to_t=45.npz')
xi_mat = xi_data['xi_mat']

outfile  = File("./results/xi_vecs_grid_32_decor_t_32min_t=27_to_t=45.pvd")

# print(xi.dat.data.shape)

n_xi = 51 # no. of xi you want to print

for i in np.arange(n_xi):

    print(f'saving xi {i+1}')
    print("Local time:",time.strftime("%H:%M:%S", time.localtime()))
    
    xi.assign(0)
    xi.dat.data[:] = xi_mat[i,:]

    xi.rename("xi")

    outfile.write(xi)

