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

Nx = 7*32
Ny = 32
mesh = PeriodicRectangleMesh(Nx, Ny, 7, 1, direction="x")

V1 = VectorFunctionSpace(mesh, "CG", 1)
V2 = FunctionSpace(mesh, "CG", 1)
V0 = FunctionSpace(mesh, "DG", 0)

u_ou = Function(V1) ; theta_ou = Function(V2)  # ou process fields
u_g = Function(V1)  ; theta_g = Function(V2)   # gaussian noise fields
u_ad = Function(V1) ; theta_ad = Function(V2)  # adapted sol. fields

u_ou.assign(0) ; theta_ou.assign(0)
u_g.assign(0)  ; theta_g.assign(0)
u_ad.assign(0) ; theta_ad.assign(0)


current_time = time.strftime("%H:%M:%S", time.localtime())
print("Local time at the start of simulation:",current_time)


def pattern_correlation(f_truth, f):
    """ compute the relative pattern correlation between two sacaler valued functions. See Say's paper for definition.
    :arg f_truth: the function against which we want the comparison; coarse grained truth
    :arg f: the deterministic or the adapted reference solution
    """
    return assemble( inner(f, f_truth)*dx)/ np.sqrt(assemble(inner(f, f)*dx) * assemble(inner(f_truth, f_truth)*dx))

def relative_l2_error(f_truth, f):
    """ compute the relative L2 error between two functions. See Wei's paper for definition.
    :arg f_truth: the function against which we want the comparison; coarse grained truth
    :arg f: the deterministic or the adapted reference solution
    """
    return errornorm(f_truth, f)/norm(f_truth)

pc_vort_ou_v_ad_global = [] # pattern correlation OU process vs adapted reference solution for all time steps
pc_vort_gauss_v_ad_global = [] # pattern correlation gaussian noise vs adapted reference solution for all time steps

l2_vel_ou_v_ad_global = [] # relative l2 error between OU process and adapted reference solution 
l2_vel_gauss_v_ad_global = [] # relative l2 error between Gaussian process and adapted reference solution

l2_temp_ou_v_ad_global = []
l2_temp_gauss_v_ad_global = []

l2_vort_ou_v_ad_global = []
l2_vort_gauss_v_ad_global = []

n_particles = 50 # no. of particles in the ensemble; this depends on the no. of particles for which data is available

for i in t_stamps:

    print(f'loading the velocity and temperature fields at t = {i}')

    pc_vort_ou_v_ad_loc = [] # pattern correlation OU process vs adapted reference solution for one time step
    pc_vort_gauss_v_ad_loc = [] # pattern correlation gaussian noise vs adapted reference solution for one time step
    
    l2_vel_ou_v_ad_loc = [] # relative l2 error between  OU process and  adapted reference solution for one time step
    l2_vel_gauss_v_ad_loc = [] # relative l2 error between  Gaussian noise and  adapted reference solution for one time step
    
    l2_temp_ou_v_ad_loc = []
    l2_temp_gauss_v_ad_loc = []
    
    l2_vort_ou_v_ad_loc = []
    l2_vort_gauss_v_ad_loc = []

    # loading data from adapted ref. sol. at one time instance 
    data_ad = np.load('../../../deterministic/224x32/global_comparison/coarse_grained_fields_data/u_theta_adapted_sol_arrays_mesh_32_at_t_'+str(i)+'.npz')
    
    vel_array_ad = data_ad['vel_array'] ; temp_array_ad = data_ad['temp_array']
    u_ad.dat.data[:] = vel_array_ad ; theta_ad.dat.data[:] = temp_array_ad
    vort_ad= interpolate(u_ad[1].dx(0) - u_ad[0].dx(1), V0)
    
    # going through each ensemble member and calculating l2 error and pattern correlation 
    for j in range(n_particles):
    
        # loading data for each particle corresponding to time t = i
        data_ou = np.load('../ensemble_vel_temp_data_as_arrays/OU_sim_var_90_mesh_32_particle_'+str(j)+'fields_data_at_t_'+str(i)+'.npz')
        data_gauss = np.load('../ensemble_vel_temp_data_as_arrays/gaussian_sim_var_90_mesh_32_particle_'+str(j)+'fields_data_at_t_'+str(i)+'.npz')

        vel_array_ou = data_ou['vel_array'] ; temp_array_ou = data_ou['temp_array']
        vel_array_gauss = data_gauss['vel_array'] ; temp_array_gauss = data_gauss['temp_array']
        

        u_ou.dat.data[:] = vel_array_ou ; theta_ou.dat.data[:] = temp_array_ou
        u_g.dat.data[:] = vel_array_gauss ; theta_g.dat.data[:] = temp_array_gauss

        # calculating vorticity from velocity !
        vort_ou= interpolate(u_ou[1].dx(0) - u_ou[0].dx(1), V0)
        vort_g= interpolate(u_g[1].dx(0) - u_g[0].dx(1), V0)
        

        ### calculating pattern correlation; OU vs adapted ref. sol.
        pc_vort_ou_v_ad_loc.append(pattern_correlation(vort_ad, vort_ou))

        ### calculating pattern correlation; gaussian noise vs adapted ref. sol.
        pc_vort_gauss_v_ad_loc.append(pattern_correlation(vort_ad, vort_g))

        ### calculating relative l2 errors between OU and adapted ref. sol.
        l2_vel_ou_v_ad_loc.append(relative_l2_error(u_ad, u_ou))
        l2_temp_ou_v_ad_loc.append(relative_l2_error(theta_ad, theta_ou))
        l2_vort_ou_v_ad_loc.append(relative_l2_error(vort_ad, vort_ou))

        ### calculating relative l2 errors between Gaussian noise and adapted ref. sol.
        l2_vel_gauss_v_ad_loc.append(relative_l2_error(u_ad, u_g))
        l2_temp_gauss_v_ad_loc.append(relative_l2_error(theta_ad, theta_g))
        l2_vort_gauss_v_ad_loc.append(relative_l2_error(vort_ad, vort_g))

    # collecting data from all ensemble members into one array
    pc_vort_ou_v_ad_global.append(pc_vort_ou_v_ad_loc)
    pc_vort_gauss_v_ad_global.append(pc_vort_gauss_v_ad_loc)

    l2_vel_ou_v_ad_global.append(l2_vel_ou_v_ad_loc)
    l2_temp_ou_v_ad_global.append(l2_temp_ou_v_ad_loc)
    l2_vort_ou_v_ad_global.append(l2_vort_ou_v_ad_loc)

    l2_vel_gauss_v_ad_global.append(l2_vel_gauss_v_ad_loc)
    l2_temp_gauss_v_ad_global.append(l2_temp_gauss_v_ad_loc)
    l2_vort_gauss_v_ad_global.append(l2_vort_gauss_v_ad_loc)

    data_file = './pattern_correlation_l2_error_data/ou_gaussian_v_adap_pattern_corr_l2_error_t27_to_45.npz'
    np.savez(data_file, pc_vort_ou_v_ad = np.array(pc_vort_ou_v_ad_global), pc_vort_gauss_v_ad = np.array(pc_vort_gauss_v_ad_global),
             l2_vel_ou_v_ad = np.array(l2_vel_ou_v_ad_global), l2_temp_ou_v_ad = np.array(l2_temp_ou_v_ad_global), 
             l2_vort_ou_v_ad = np.array(l2_vort_ou_v_ad_global), l2_vel_gauss_v_ad = np.array(l2_vel_gauss_v_ad_global),
             l2_temp_gauss_v_ad = np.array(l2_temp_gauss_v_ad_global), l2_vort_gauss_v_ad = np.array(l2_vort_gauss_v_ad_global))


print("Local time at the end of simulation:",time.strftime("%H:%M:%S", time.localtime()))

