import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from firedrake import *
import math
import time
from firedrake.petsc import PETSc
from utilities import OU_mat

my_ensemble = Ensemble(COMM_WORLD, 1)
spatial_comm = my_ensemble.comm
ensemble_comm = my_ensemble.ensemble_comm

PETSc.Sys.Print(f'size of ensemble is {ensemble_comm.size}')
PETSc.Sys.Print(f'size of spatial communicators is {spatial_comm.size}')

Nx = 7*32
Ny = 32
mesh = PeriodicRectangleMesh(Nx, Ny, 7, 1, direction="x", comm = spatial_comm)

data = np.load('../vel_temp_data/u_theta_coarse_grained_arrrays.npz')

vel_array = data['vel_array']
temp_array = data['temp_array']

V1 = VectorFunctionSpace(mesh, "CG", 1)
V2 = FunctionSpace(mesh, "CG", 1)
V0 = FunctionSpace(mesh, "DG", 0)

x, y = SpatialCoordinate(mesh)

# define dimensionless parameters
Ro = 0.3 ; Re = 10**4 ; C = 0.02 ; Pe = 10**4

Z = V1*V2

utheta = Function(Z)
u, theta = split(utheta)
v, phi = TestFunctions(Z)
u_ = Function(V1)
theta_ = Function(V2)
u_pert = Function(V1) # function to hold stochastic term

u_.assign(0)
theta_.assign(0)
u_.dat.data[:] = vel_array
theta_.dat.data[:] = temp_array
# vort_= interpolate(u_[1].dx(0) - u_[0].dx(1), V0)


perp = lambda arg: as_vector((-arg[1], arg[0]))

Dt = 0.04 # 32.4 minutes

F = ( inner(u-u_,v)
    + Dt*0.5*(inner(dot(u, nabla_grad(u)), v) + inner(dot(u_, nabla_grad(u_)), v))
    + Dt*0.5*(1/Ro)*inner(perp(u) + perp(u_), v)
    - Dt*0.5*(1/C)*(theta + theta_)* div(v)
    + Dt *0.5 *(1/Re)*inner((nabla_grad(u)+nabla_grad(u_)), nabla_grad(v))
    + np.sqrt(Dt)*0.5*(inner(dot(u_pert, nabla_grad(u)), v) + inner(dot(u_pert, nabla_grad(u_)), v))
    + np.sqrt(Dt)*0.5*(inner((u_[0]+u[0])*grad(u_pert[0]) + (u_[1]+u[1])*grad(u_pert[1]) , v))
    + (theta - theta_)*phi 
    - Dt*0.5*inner(theta_*u_ + theta*u, grad(phi))
    + Dt*0.5*(1/Pe)*inner((grad(theta)+grad(theta_)),grad(phi)) 
    + np.sqrt(Dt)*0.5*inner(u_pert, grad(theta) + grad(theta_))*phi)*dx

bound_cond = [DirichletBC(Z.sub(0).sub(1), Constant(0.0), (1,2))]


xi_data = np.load('../../../deterministic/224x32/xi_calculation_visualization/xi_vec_data/xi_matrix_combined_calc_51_eigvec_grid_32_c_1_by_32_decor_t_32min_t=27_to_t=45.npz')
xi_mat = xi_data['xi_mat']
PETSc.Sys.Print(f'loaded the xi matrix for rank {ensemble_comm.rank}, local time: {time.strftime("%H:%M:%S", time.localtime())}')

delta_x = 7/4
delta_y = 1/4
n = 3
gridpoints = np.array([[delta_x + i * delta_x, delta_y + j * delta_y] for j in range(n) for i in range(n)])

t_start = 27.0 + Dt # day 15
t_end = 45.0 # day 25

seed_no = ensemble_comm.rank 
PETSc.Sys.Print(f'seed_no assigned for rank {ensemble_comm.rank} is {seed_no}, local time: {time.strftime("%H:%M:%S", time.localtime())}')

n_t_steps = int((t_end - t_start)/Dt)
n_EOF = 10 # desired no. of EOFs to include in stochastic sim
##############
# 51 EOFs ---------> 90 % variance
# 21 EOFs ---------> 70 % variance
# 10 EOFs ---------> 50 % variance 
##############
acf_data = np.load('../../../deterministic/224x32/acf_data/acf_1_data_51_EOFs_mesh_32.npz')
acf1_data = acf_data['acf1_data'] # this is an array containing the lag 1 ACF for the SVD time-series data corresponding to each xi

PETSc.Sys.Print(f'loaded the acf1 data corresponding to time-series of first {acf1_data.size} xi')

np.random.seed(seed_no)
rand_mat = OU_mat(n_t_steps+2, n_EOF, acf1_data) # time-series generated from OU process
# rand_mat = np.random.normal(size=(n_t_steps+2, n_EOF)) # time-series generated from Gaussian process

data_file = '../ensemble_vel_temp_data_as_arrays/OU_sim_var_50_mesh_32_particle_'+str(seed_no)+'fields_data_at_t_27.0.npz'
np.savez(data_file, vel_array = u_.dat.data, temp_array = theta_.dat.data)

t = 27.0 + Dt
iter_n = 1
freq = 25
big_t_step = freq*Dt 
current_time = time.strftime("%H:%M:%S", time.localtime())
PETSc.Sys.Print(f'Local time at the start of simulation for particle {seed_no}: {time.strftime("%H:%M:%S", time.localtime())}')
start_time = time.time()

PETSc.Sys.Print(f'particle no:{seed_no}, local time:{round(t,4)}')

while (round(t,4) <= t_end):
    vec_u_pert = np.zeros((xi_mat.shape[1], 2))
    for i in range(n_EOF):
        vec_u_pert +=  rand_mat[iter_n-1,i]*xi_mat[i, :,:]

    u_pert.assign(0)
    u_pert.dat.data[:] = vec_u_pert

    solve(F == 0, utheta, bcs = bound_cond)
    u, theta = utheta.subfunctions
    if iter_n%freq == 0:
        if iter_n == freq:
            end_time = time.time()
            execution_time = (end_time-start_time)/60 # running time for one time step (t_step)
            PETSc.Sys.Print(f'approx. running time for one big t_step for particle {seed_no} is {round(execution_time,2)} minutes')
            total_execution_time = ((t_end - t_start)/big_t_step)*execution_time
            PETSc.Sys.Print(f'approx. total running time for particle {seed_no} is {round(total_execution_time,2)} minutes')

        PETSc.Sys.Print(f'particle no:{seed_no}, simulation time:{round(t,4)}, local time: {time.strftime("%H:%M:%S", time.localtime())}')
        # vort = interpolate(u[1].dx(0) - u[0].dx(1), V0)

        data_file = '../ensemble_vel_temp_data_as_arrays/OU_sim_var_50_mesh_32_particle_'+str(seed_no)+'fields_data_at_t_'+str(round(t,4))+'.npz'
        np.savez(data_file, vel_array = u.dat.data, temp_array = theta.dat.data)

    
    u_.assign(u)
    theta_.assign(theta)

    t += Dt
    iter_n +=1

print(f'Local time at the end of simulation for particle {seed_no} is {time.strftime("%H:%M:%S", time.localtime())}')
