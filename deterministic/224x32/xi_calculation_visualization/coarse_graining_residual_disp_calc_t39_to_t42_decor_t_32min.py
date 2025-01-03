import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from firedrake import *
import math
import time

dX = []
vel_data_truth, temp_data_truth, vort_data_truth = [], [], []

Dt_uc = 0.04 # assumed decorrelated time (same as Dt)
t_start = 39
t_end = 42
time_array = np.arange(t_start, t_end, Dt_uc)
t_stamps = np.round(time_array, 2)

print("time_stamps:", time_array)

Nx = 7*32
Ny = 32
c_mesh = PeriodicRectangleMesh(Nx, Ny, 7, 1, direction="x")
c_mesh.name = "coarse_mesh"

V1c = VectorFunctionSpace(c_mesh, "CG", 1)
V2c = FunctionSpace(c_mesh, "CG", 1)
V0c = FunctionSpace(c_mesh, "DG", 0)

coords_func_coarse = Function(V1c).interpolate(SpatialCoordinate(c_mesh))
coords_coarse = coords_func_coarse.dat.data

uc = Function(V1c)
thetac = Function(V2c)

Dt = 0.04 # coarse grid time step

n_iter = 0
for i in t_stamps:
    
    print('time:', i)
    print("loading fine resolution mesh and velocity.....",
      "current time:",time.strftime("%H:%M:%S", time.localtime()))
    
    with CheckpointFile("../../1792x256/h5_files/grid_256_fields_at_time_t="+str(i)+".h5", 'r') as afile:
        mesh = afile.load_mesh("mesh_256")
        u_ = afile.load_function(mesh, "velocity") 
        theta_ = afile.load_function(mesh, "temperature")
    
    print("finished loading! Now getting value of loaded function on coord.... ",
      time.strftime("%H:%M:%S", time.localtime()))
    
    V1f = VectorFunctionSpace(mesh, "CG", 1)
    V2f = FunctionSpace(mesh, "CG", 1)
    V0f = FunctionSpace(mesh, "DG", 0)

    print("Coarse graining.........",
        time.strftime("%H:%M:%S", time.localtime()))

    #####Averaging and Coarse graining#########
    u_trial = TrialFunction(V1f)
    u_test = TestFunction(V1f)
    u_avg = Function(V1f)
    
    c_sqr = Constant(1/(32**2)) # averaging solution within box of size 1/32x1/32

    a_vel = (c_sqr * inner(nabla_grad(u_trial), nabla_grad(u_test)) + inner(u_trial, u_test)) * dx
    l_vel = inner(u_, u_test) * dx

    # step 1: spatial averaging using Helmholtz operator
    bound_cond = [DirichletBC(V1f.sub(1), Constant(0.0), (1,2))] # making sure that n.v is zero after coarse graining

    solve(a_vel==l_vel, u_avg, bcs = bound_cond)

    print("solved the vel PDE (alpha-regularization)",
        time.strftime("%H:%M:%S", time.localtime()))

    # projecting on coarse grid

    print("retrieving velocity data........",
        time.strftime("%H:%M:%S", time.localtime()))
    uc.assign(0)
    u_avg_vals = np.array(u_avg.at(coords_coarse, tolerance=1e-10))
    uc.dat.data[:] = u_avg_vals

    print("calculating (u - u_avg)Dt........",
        time.strftime("%H:%M:%S", time.localtime()))

    dX.append(Dt*(np.array(u_.at(coords_coarse, tolerance=1e-10)) 
                - np.array(uc.at(coords_coarse, tolerance=1e-10))))
    
    print("Calculation done, saving the data into a separate file",
        time.strftime("%H:%M:%S", time.localtime()))
    
    dX_x = np.array(dX)[:,:,0]
    print("shape of dX1_x:", dX_x.shape)

    dX_y = np.array(dX)[:,:,1]
    print("shape of dX1_y:", dX_y.shape)

    data_file_1 = './data_for_xi_calculation/dX_data_t='+str(t_start)+'_to_t='+str(t_end)+'_grid_32_c=1by32_decor_t_32min.npz'
    np.savez(data_file_1, dX_x = dX_x, dX_y = dX_y)

print("simulation completed !!!", time.strftime("%H:%M:%S", time.localtime()))