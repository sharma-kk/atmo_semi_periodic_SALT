import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from firedrake import *
import math
import time

delta_x = 7/4
delta_y = 1/4
n = 3
vel_data, temp_data, vort_data = [], [], []

gridpoints = np.array([[delta_x + i * delta_x, delta_y + j * delta_y] for j in range(n) for i in range(n)])

print("loading coarse grained velocity and temp......",
    "current time:",time.strftime("%H:%M:%S", time.localtime()))

with CheckpointFile("../1792x256/h5_files/coarse_grained_vel_temp_at_t=27.0_mesh_64_c_1_by_64.h5", 'r') as afile:
    mesh = afile.load_mesh("coarse_mesh")
    uc = afile.load_function(mesh, "coarse_vel") 
    thetac = afile.load_function(mesh, "coarse_temp")

print("finished loading! calculating vorticity from velocity",
    time.strftime("%H:%M:%S", time.localtime()))

V1 = VectorFunctionSpace(mesh, "CG", 1)
V2 = FunctionSpace(mesh, "CG", 1)
V0 = FunctionSpace(mesh, "DG", 0)

vort_= interpolate(uc[1].dx(0) - uc[0].dx(1), V0)
vort_.rename("vorticity")

x, y = SpatialCoordinate(mesh)
# define dimensionless parameters
Ro = 0.3 ; Re = 3*10**4 ; B = 0 ; C = 0.02 ; Pe = 3*10**4

Z = V1*V2

utheta = Function(Z)
u, theta = split(utheta)
v, phi = TestFunctions(Z)
u_ = Function(V1)
theta_ = Function(V2)

u_.assign(uc)
theta_.assign(thetac)

vel_data.append(np.array(u_.at(gridpoints, tolerance=1e-10)))
temp_data.append(np.array(theta_.at(gridpoints, tolerance=1e-10)))
vort_data.append(np.array(vort_.at(gridpoints, tolerance=1e-10)))
   
perp = lambda arg: as_vector((-arg[1], arg[0]))

Dt =0.02 # 16.2 minutes

f_stoch = Function(V1) # we apply this instead of SALT forcing
f_stoch_ = Function(V1)

F = ( inner(u-u_,v)
    + Dt*0.5*(inner(dot(u, nabla_grad(u)), v) + inner(dot(u_, nabla_grad(u_)), v))
    + Dt*0.5*(1/Ro)*inner((perp(u) + perp(u_)), v)
    - Dt*0.5*(1/C)*(theta + theta_)* div(v)
    + Dt *0.5 *(1/Re)*inner((nabla_grad(u)+nabla_grad(u_)), nabla_grad(v))
    + np.sqrt(Dt)*0.5*(inner(dot(f_stoch, nabla_grad(u)), v) + inner(dot(f_stoch_, nabla_grad(u_)), v))
    + np.sqrt(Dt)*0.5*(inner(u[0]*grad(f_stoch[0]) + u_[0]*grad(f_stoch_[0]), v))
    + np.sqrt(Dt)*0.5*(inner(u[1]*grad(f_stoch[1]) + u_[1]*grad(f_stoch_[1]), v))
    + (theta - theta_)*phi 
    - Dt*0.5*inner(theta_*u_ + theta*u, grad(phi))
    + Dt*0.5*(1/Pe)*inner((grad(theta) + grad(theta_)),grad(phi)) 
    + np.sqrt(Dt)*0.5*(inner(f_stoch, grad(theta)) + inner(f_stoch_, grad(theta_)))*phi)*dx

bound_cond = [DirichletBC(Z.sub(0).sub(1), Constant(0.0), (1,2))]

# visulization at t=27.0
theta_.rename("temperature")
u_.rename("velocity")
energy_ = 0.5*(norm(u_)**2)
KE = []
KE.append(energy_)
print(f'KE at time t=27.0: {round(energy_,6)}')

outfile = File("./results/adaptive_ref_sol_grid_64_t27_to_t45.pvd")
outfile.write(u_, theta_, vort_)

# time stepping and visualization at other time steps
t_start = 27.0 + Dt # day 15
t_end = 45 # day 25

###### loading the stochastic force matrix ####
stoch_f_data = np.load('./forcing_adap_ref_sol/forcing/stochastic_forcing_as_deterministic_t27_to_t45_mesh64_var90.npz') # forcing for 90% variance
stoch_f_mat = stoch_f_data['stoch_f_mat']
print(f'shape of stochastic forcing matrix: {stoch_f_mat.shape}')

t = 27.0 + Dt
iter_n = 1
freq = 50
t_step = freq*Dt # 1 time unit
current_time = time.strftime("%H:%M:%S", time.localtime())
print("Local time at the start of simulation:",current_time)
start_time = time.time()
data_file_2 = "./KE_details/adaptive_ref_sol_grid_64_t27_to_t45.txt"

while (round(t,4) <= t_end):
    f_stoch_.assign(0)
    f_stoch.assign(0)
    f_stoch_.dat.data[:] = stoch_f_mat[iter_n-1,:,:]
    f_stoch.dat.data[:] = stoch_f_mat[iter_n,:,:]
    solve(F == 0, utheta, bcs = bound_cond)
    u, theta = utheta.subfunctions
    if iter_n%freq == 0:
        if iter_n == freq:
            end_time = time.time()
            execution_time = (end_time-start_time)/60 # running time for one time step (t_step)
            print("Approx. running time for one t_step: %.2f minutes" %execution_time)
            total_execution_time = ((t_end - t_start)/t_step)*execution_time
            print("Approx. total running time: %.2f minutes:" %total_execution_time)

        print("t=", round(t,4))

        energy = 0.5*(norm(u)**2)
        KE.append(energy)
        print("kinetic energy:", round(KE[-1],6))
        with open(data_file_2, 'w') as ff:
            print(f'KE_over_time = {KE}', file = ff)

        vort = interpolate(u[1].dx(0) - u[0].dx(1), V0)

        vel_data.append(np.array(u.at(gridpoints, tolerance=1e-10)))
        temp_data.append(np.array(theta.at(gridpoints, tolerance=1e-10)))
        vort_data.append(np.array(vort.at(gridpoints, tolerance=1e-10)))

        data_file_1 = './data_from_deterministic_run/adaptive_sol_mesh_64_sim_vel_temp_vort_data_t=27_to_t=45_coarse_grained_ic.npz'
        np.savez(data_file_1, gridpoints = gridpoints, vel_data = np.array(vel_data), temp_data= np.array(temp_data), vort_data = np.array(vort_data))

        vort.rename("vorticity")
        theta.rename("temperature")
        u.rename("velocity")
        outfile.write(u, theta, vort)
    u_.assign(u)
    theta_.assign(theta)

    t += Dt
    iter_n +=1

print("Local time at the end of simulation:",time.strftime("%H:%M:%S", time.localtime()))