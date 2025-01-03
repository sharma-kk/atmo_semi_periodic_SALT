import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from firedrake import *
import math
import time

time_delta = 1.0 # saving the fields at an interval of time_delta
t_start = 27
t_end = 45
time_array = np.arange(t_start, t_end + 1, time_delta)
t_stamps = np.round(time_array, 1)

print("time_stamps:", time_array)

Nx = 7*64
Ny = 64
c_mesh = PeriodicRectangleMesh(Nx, Ny, 7, 1, direction="x")
c_mesh.name = "coarse_mesh"

V1c = VectorFunctionSpace(c_mesh, "CG", 1)
V2c = FunctionSpace(c_mesh, "CG", 1)
V0c = FunctionSpace(c_mesh, "DG", 0)

coords_func_coarse = Function(V1c).interpolate(SpatialCoordinate(c_mesh))
coords_coarse = coords_func_coarse.dat.data

uc = Function(V1c)
thetac = Function(V2c)

n_iter = 0

outfile = File('../results/coarse_grained_fields_grid_64_t27_to_t45.pvd')

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
    
    c_sqr = Constant(1/(64**2)) # averaging solution within box of size 1/32x1/32

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

    print("solving the temp PDE (alpha-regularization)",
    time.strftime("%H:%M:%S", time.localtime()))

    theta_trial = TrialFunction(V2f)
    theta_test = TestFunction(V2f)
    theta_avg = Function(V2f)
    a_temp = (c_sqr * inner(grad(theta_trial), grad(theta_test)) + theta_trial*theta_test) * dx
    l_temp = theta_*theta_test* dx

    solve(a_temp==l_temp, theta_avg)
    
    print("retrieving temperature data........",
        time.strftime("%H:%M:%S", time.localtime()))
    thetac.assign(0)
    theta_avg_vals = np.array(theta_avg.at(coords_coarse, tolerance=1e-10))
    thetac.dat.data[:] = theta_avg_vals

    uc.rename("coarse_vel")
    thetac.rename("coarse_temp")

    outfile.write(uc, thetac)

    data_file ='./fields_as_arrays/u_theta_coarse_grained_arrays_mesh_64_at_t_'+str(i)+'.npz'
    np.savez(data_file, vel_array = uc.dat.data, temp_array = thetac.dat.data)

print("simulation completed !!!", time.strftime("%H:%M:%S", time.localtime()))