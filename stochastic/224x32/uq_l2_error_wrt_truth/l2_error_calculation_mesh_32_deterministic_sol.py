import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from firedrake import *
import math
import time

grid = 32
print(f'grid: {grid}')

Nx = 7*grid
Ny = grid
mesh = PeriodicRectangleMesh(Nx, Ny, 7, 1, direction="x")

time_delta = 1.0 # saving the fields at an interval of time_delta
t_start = 27
t_end = 45
time_array = np.arange(t_start, t_end + 1, time_delta)
t_stamps = np.round(time_array, 1)

print("time_stamps:", time_array)

V1 = VectorFunctionSpace(mesh, "CG", 1)
V2 = FunctionSpace(mesh, "CG", 1)
V0 = FunctionSpace(mesh, "DG", 0)

u_d = Function(V1) ; theta_d = Function(V2)  # deterministic solution
u_t = Function(V1) ; theta_t = Function(V2)  # coarse grained fields

u_d.assign(0) ; theta_d.assign(0)
u_t.assign(0) ; theta_t.assign(0)


current_time = time.strftime("%H:%M:%S", time.localtime())
print("Local time at the start of simulation:",current_time)


def relative_l2_error(f_truth, f):
    """ compute the relative L2 error between two functions. See Wei's paper for definition.
    :arg f_truth: the function against which we want the comparison; coarse grained truth
    :arg f: the deterministic or the adapted reference solution
    """
    return errornorm(f_truth, f)/norm(f_truth)


l2_vel_det_v_truth_global = [] # relative l2 error between deterministic and coarse grained truth  
l2_temp_det_v_truth_global = []

for i in t_stamps:

    print(f'loading the velocity and temperature fields at t = {i}')

    # loading data from coarse grained solution at one time instance 
    data_t = np.load('../../../deterministic/224x32/global_comparison/coarse_grained_fields_data/u_theta_coarse_grained_arrrays_mesh_32_at_t_'+str(i)+'.npz')
    
    vel_array_t = data_t['vel_array'] ; temp_array_t = data_t['temp_array']
    u_t.dat.data[:] = vel_array_t ; theta_t.dat.data[:] = temp_array_t

    # loading data from deterministic solution at one time instance 
    data_d = np.load('../../../deterministic/224x32/global_comparison/coarse_grained_fields_data/u_theta_deter_sim_arrays_mesh_32_at_t_'+str(i)+'.npz')
    
    vel_array_d = data_d['vel_array'] ; temp_array_d = data_d['temp_array']
    u_d.dat.data[:] = vel_array_d ; theta_d.dat.data[:] = temp_array_d


    l2_vel_det_v_truth_global.append(relative_l2_error(u_t, u_d))
    l2_temp_det_v_truth_global.append(relative_l2_error(theta_t, theta_d))


    data_file = './l2_error_data/det_v_truth_grid_'+str(grid)+'_l2_error_t27_to_45.npz'
    np.savez(data_file, l2_vel_det_v_truth = np.array(l2_vel_det_v_truth_global), l2_temp_det_v_truth = np.array(l2_temp_det_v_truth_global))


print("Local time at the end of simulation:",time.strftime("%H:%M:%S", time.localtime()))

