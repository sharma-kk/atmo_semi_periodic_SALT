import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from firedrake import *
import math
import time

print("loading coarse grained velocity and temp......",
    "current time:",time.strftime("%H:%M:%S", time.localtime()))

with CheckpointFile("../h5_files/coarse_grained_vel_temp_at_t=27.0_mesh_64_c_1_by_64.h5", 'r') as afile:
    mesh = afile.load_mesh("coarse_mesh")
    uc = afile.load_function(mesh, "coarse_vel") 
    thetac = afile.load_function(mesh, "coarse_temp")

print("finished loading! calculating vorticity from velocity",
    time.strftime("%H:%M:%S", time.localtime()))

V1 = VectorFunctionSpace(mesh, "CG", 1)
V2 = FunctionSpace(mesh, "CG", 1)
V0 = FunctionSpace(mesh, "DG", 0)

x, y = SpatialCoordinate(mesh)
# define dimensionless parameters ; grid 64
Ro = 0.3 ; Re = 3*10**4 ; B = 0 ; C = 0.02 ; Pe = 3*10**4

Z = V1*V2

utheta = Function(Z)
u, theta = split(utheta)
v, phi = TestFunctions(Z)
u_ = Function(V1)
theta_ = Function(V2)

u_.assign(uc)
theta_.assign(thetac)

perp = lambda arg: as_vector((-arg[1], arg[0]))

Dt =0.02 # 16.2 minutes

F = ( inner(u-u_,v)
    + Dt*0.5*(inner(dot(u, nabla_grad(u)), v) + inner(dot(u_, nabla_grad(u_)), v))
    + Dt*0.5*(1/Ro)*inner((perp(u) + perp(u_)), v)
    - Dt*0.5*(1/C)*(theta + theta_)* div(v)
    + Dt *0.5 *(1/Re)*inner((nabla_grad(u)+nabla_grad(u_)), nabla_grad(v))
    + (theta - theta_)*phi - Dt*0.5*inner(theta_*u_ + theta*u, grad(phi))
    + Dt*0.5*(1/Pe)*inner((grad(theta)+grad(theta_)),grad(phi)) )*dx


bound_cond = [DirichletBC(Z.sub(0).sub(1), Constant(0.0), (1,2))]


# saving the fields as arrays at t = 27
data_file ='./fields_as_arrays/u_theta_deter_sim_arrays_mesh_64_at_t_27.0.npz'
np.savez(data_file, vel_array = u_.dat.data, temp_array = theta_.dat.data)

# time stepping and visualization at other time steps
t_start = 27.0 + Dt # day 15
t_end = 45.0 # day 25

t = 27.0 + Dt
iter_n = 1
freq = 50
t_step = freq*Dt 
current_time = time.strftime("%H:%M:%S", time.localtime())
print("Local time at the start of simulation:",current_time)
start_time = time.time()

while (round(t,4) <= t_end):
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

        data_file ='./fields_as_arrays/u_theta_deter_sim_arrays_mesh_64_at_t_'+str(round(t,4))+'.npz'
        np.savez(data_file, vel_array = u.dat.data, temp_array = theta.dat.data)

    u_.assign(u)
    theta_.assign(theta)

    t += Dt
    iter_n +=1

print("Local time at the end of simulation:",time.strftime("%H:%M:%S", time.localtime()))