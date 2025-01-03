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

u_cg = Function(V1) ; theta_cg = Function(V2) # coarse grained fields
u_det = Function(V1) ; theta_det = Function(V2)  # deterministic sol. fields
u_ad = Function(V1) ; theta_ad = Function(V2)  # adapted sol. fields

u_cg.assign(0) ; theta_cg.assign(0)
u_det.assign(0) ; theta_det.assign(0)
u_ad.assign(0) ; theta_ad.assign(0)

# outfile = File("./results/deterministic_adapted_coarse_grained_fields_grid_32_t27_to_t45.pvd")

current_time = time.strftime("%H:%M:%S", time.localtime())
print("Local time at the start of simulation:",current_time)

def relative_l2_error(f_truth, f):
    """ compute the relative L2 error between two functions. See Wei's paper for definition.
    :arg f_truth: the function against which we want the comparison; coarse grained truth
    :arg f: the deterministic or the adapted reference solution
    """
    return errornorm(f_truth, f)/norm(f_truth)

def pattern_correlation(f_truth, f):
    """ compute the relative pattern correlation between two sacaler valued functions. See Say's paper for definition.
    :arg f_truth: the function against which we want the comparison; coarse grained truth
    :arg f: the deterministic or the adapted reference solution
    """
    return assemble( inner(f, f_truth)*dx)/ np.sqrt(assemble(inner(f, f)*dx) * assemble(inner(f_truth, f_truth)*dx))

l2_error_temp_det_vs_truth = [] ; l2_error_vort_det_vs_truth = [] ; l2_error_vel_det_vs_truth = []
l2_error_temp_adap_vs_truth = [] ; l2_error_vort_adap_vs_truth = [] ; l2_error_vel_adap_vs_truth = []

pattern_corr_temp_det_vs_truth = [] ; pattern_corr_vort_det_vs_truth = []
pattern_corr_temp_adap_vs_truth = [] ; pattern_corr_vort_adap_vs_truth = []

for i in t_stamps:

    print(f'loading the velocity and temperature fields at t = {i}')

    data_cg = np.load('./coarse_grained_fields_data/u_theta_coarse_grained_arrrays_mesh_32_at_t_'+str(i)+'.npz')
    data_det = np.load('./coarse_grained_fields_data/u_theta_deter_sim_arrays_mesh_32_at_t_'+str(i)+'.npz')
    data_ad = np.load('./coarse_grained_fields_data/u_theta_adapted_sol_arrays_mesh_32_at_t_'+str(i)+'.npz')

    vel_array_cg = data_cg['vel_array'] ; temp_array_cg = data_cg['temp_array']
    vel_array_det = data_det['vel_array'] ; temp_array_det = data_det['temp_array']
    vel_array_ad = data_ad['vel_array'] ; temp_array_ad = data_ad['temp_array']

    u_cg.dat.data[:] = vel_array_cg;  theta_cg.dat.data[:] = temp_array_cg
    u_det.dat.data[:] = vel_array_det;  theta_det.dat.data[:] = temp_array_det
    u_ad.dat.data[:] = vel_array_ad;  theta_ad.dat.data[:] = temp_array_ad

    vort_cg= interpolate(u_cg[1].dx(0) - u_cg[0].dx(1), V0)
    vort_det= interpolate(u_det[1].dx(0) - u_det[0].dx(1), V0)
    vort_ad= interpolate(u_ad[1].dx(0) - u_ad[0].dx(1), V0)

    ### calculating the l2 error and pattern correlation; deterministic vs truth
    l2_error_temp_det_vs_truth.append(relative_l2_error(theta_cg, theta_det))
    l2_error_vort_det_vs_truth.append(relative_l2_error(vort_cg, vort_det))
    l2_error_vel_det_vs_truth.append(relative_l2_error(u_cg, u_det))
    pattern_corr_temp_det_vs_truth.append(pattern_correlation(theta_cg, theta_det))
    pattern_corr_vort_det_vs_truth.append(pattern_correlation(vort_cg, vort_det))

    ### calculating the l2 error and pattern correlation; adapted sol. vs truth
    l2_error_temp_adap_vs_truth.append(relative_l2_error(theta_cg, theta_ad))
    l2_error_vort_adap_vs_truth.append(relative_l2_error(vort_cg, vort_ad))
    l2_error_vel_adap_vs_truth.append(relative_l2_error(u_cg, u_ad))
    pattern_corr_temp_adap_vs_truth.append(pattern_correlation(theta_cg, theta_ad))
    pattern_corr_vort_adap_vs_truth.append(pattern_correlation(vort_cg, vort_ad))

    # u_cg.rename('vel_truth') ; theta_cg.rename('temp_truth')
    # u_det.rename('vel_det') ; theta_det.rename('temp_det')
    # u_ad.rename('vel_adap') ; theta_ad.rename('temp_adap')

    # outfile.write(u_cg, theta_cg, u_det, theta_det, u_ad, theta_ad)

    data_file = './l2_error_pattern_cor_data/det_vs_adap_vs_truth_t27_to_t45.npz'
    np.savez(data_file, l2_temp_det_v_truth = np.array(l2_error_temp_det_vs_truth), l2_vort_det_v_truth = np.array(l2_error_vort_det_vs_truth)
             ,pc_temp_det_v_truth = np.array(pattern_corr_temp_det_vs_truth), pc_vort_det_v_truth = np.array(pattern_corr_vort_det_vs_truth)
             ,l2_temp_adap_v_truth = np.array(l2_error_temp_adap_vs_truth), l2_vort_adap_v_truth = np.array(l2_error_vort_adap_vs_truth)
             ,pc_temp_adap_v_truth = np.array(pattern_corr_temp_adap_vs_truth), pc_vort_adap_v_truth = np.array(pattern_corr_vort_adap_vs_truth)
             ,l2_vel_det_v_truth = np.array(l2_error_vel_det_vs_truth), l2_vel_adap_v_truth = np.array(l2_error_vel_adap_vs_truth))

print("Local time at the end of simulation:",time.strftime("%H:%M:%S", time.localtime()))

