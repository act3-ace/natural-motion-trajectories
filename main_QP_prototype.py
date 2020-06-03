#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mark

state = [x y z xdot ydot zdot] (position and velocity in Hill frame)
X[:,i] accesses the ith state vector

control = [Fx Fy Fz] (forces in Hill Frame)

"""

import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import random as rand 
import sys 
from parameters import SystemParameters 


##############################################################################
# Import the Desired Controller 
from controllers.simple_linear_controller import Controller 
##############################################################################

# Flags 
f_plot_option = 2


# Parameters 
dim_state = 6; 
dim_control = 3; 
sys_data = SystemParameters() 

mean_motion = sys_data.mean_motion
mass_chaser = sys_data.mass_chaser


# Initial Values
x0 = np.array([[18*(rand.random()-0.5)],  # x
               [18*(rand.random()-0.5)],  # y 
               [0],  # z
               [0],  # xdot 
               [0],  # ydot 
               [0]]) # zdot 
u0 = np.array([[0],  # Fx 
               [0],  # Fy
               [0]]) # Fz


# Define CWH Dynamics 
A = np.array([[0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1],
              [3*mean_motion**2, 0, 0, 0, 2*mean_motion, 0],
              [0, 0, 0, -2*mean_motion, 0, 0],
              [0, 0, -mean_motion**2, 0, 0, 0]])
B = np.array([[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              [1/mass_chaser, 0, 0],
              [0, 1/mass_chaser, 0],
              [0, 0, 1/mass_chaser]])

def F(x,u):
    return np.matmul(A,x)+np.matmul(B,u)


# Simulate 
T  = 15 # total simulation time [s]
Nsteps = 500 # number steps in time horizon 
t = np.linspace(0, T, Nsteps) # evenly spaced time instances in simulation horizon 
X = np.zeros([dim_state, Nsteps]) # state at each time 
U = np.zeros([dim_control, Nsteps]) # control at each time 
dt = t[1]-t[0]
X[:,0]=x0.reshape(dim_state)
controller = Controller() # initialize Controller class  

##############################################################################

import gurobipy as gp 
from gurobipy import GRB


# Parameters 
tau = 500 # time steps in planning horizon 
dt_plan = 2 # 
time_vector = np.linspace(0,(tau-1)*dt_plan, tau)

initial_state = x0.reshape(6)
goal_state = np.zeros(6)
n = mean_motion
mc = mass_chaser

# Set Ranges 
smax = 10000 # arbitrary (included bounds to speed up solver)
vmax = 10 # [m/s]
Fmax = 2 # [N]


# Initialize states 
sx = [] 
sy = [] 
vx = [] 
vy = [] 
Fx = [] 
Fy = [] 

m = gp.Model("QPTraj")


# Define variables at each of the tau timesteps 
for t in range(tau) : 
    sx.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -smax, ub = smax, name="sx"+str(t) )) 
    vx.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -vmax, ub = vmax, name="vx"+str(t) )) 
    Fx.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -Fmax, ub = Fmax, name="Fx"+str(t) )) 
    sy.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -smax, ub = smax, name="sy"+str(t) )) 
    vy.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -vmax, ub = vmax, name="vy"+str(t) )) 
    Fy.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -Fmax, ub = Fmax, name="Fy"+str(t) )) 

m.update()


# Set boundary conditions 
m.addConstr( sx[0] == initial_state[0] , "sx0")
m.addConstr( sy[0] == initial_state[1] , "sy0")
m.addConstr( vx[0] == initial_state[3] , "vx0")
m.addConstr( vy[0] == initial_state[4] , "vy0")
m.addConstr( sx[-1] == goal_state[0] , "sxf")
m.addConstr( sy[-1] == goal_state[1] , "syf")
m.addConstr( vx[-1] == goal_state[3] , "vxf")
m.addConstr( vy[-1] == goal_state[4] , "vyf")


# Set Dynamics 
for t in range(tau-1) :
    # Dynamics 
    m.addConstr( sx[t+1] == sx[t] + vx[t]*dt_plan , "Dsx_"+str(t))
    m.addConstr( sy[t+1] == sy[t] + vy[t]*dt_plan , "Dsy_"+str(t))
    # m.addConstr( vx[t+1] == vx[t] + Fx[t]*(1/mc)*dt , "Dvx_"+str(t) )
    # m.addConstr( vy[t+1] == vy[t] + Fy[t]*(1/mc)*dt , "Dvy_"+str(t) )

    m.addConstr( vx[t+1] == vx[t] + sx[t]*3*n**2*dt_plan + sy[t]*2*n*dt_plan + Fx[t]*(1/mc)*dt_plan , "Dvx_"+str(t) )
    m.addConstr( vy[t+1] == vy[t] - vx[t]*2*n*dt_plan                   + Fy[t]*(1/mc)*dt_plan , "Dvy_"+str(t) )


# Set Objective ( minimize: sum(Fx^2 + Fy^2) )
obj = Fx[0]*Fx[0] + Fy[0]*Fy[0]
for t in range(0,tau):
    obj = obj + Fx[t]*Fx[t] + Fy[t]*Fy[t]



m.setObjective(obj, GRB.MINIMIZE)
  

# Optimize and report on results 
m.optimize()
print("-----------------------------------------------------------------------------")


# Save desired trajectory 
desired_trajectory = np.zeros([4,tau]) 
desired_control = np.zeros([2,tau])

for t in range(0,tau): # TODO: find quicker way to do this 
    desired_trajectory[0,t] = m.getVarByName("sx"+str(t)).x
    desired_trajectory[1,t] = m.getVarByName("sy"+str(t)).x
    desired_trajectory[2,t] = m.getVarByName("vx"+str(t)).x
    desired_trajectory[3,t] = m.getVarByName("vy"+str(t)).x
    desired_control[0,t] = m.getVarByName("Fx"+str(t)).x
    desired_control[1,t] = m.getVarByName("Fy"+str(t)).x


X = desired_trajectory 
U = desired_control

##############################################################################



if f_plot_option == 0 : 
    # Style plot 
    marker_size = 5
    line_width = 2
    fig = plt.figure(figsize=(8,6))
    plt.grid()
    axis_font = 10
    ax_label_font = 10
    plt.xlabel("$x$", fontsize=ax_label_font)
    plt.ylabel("$y$", fontsize=ax_label_font)
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : axis_font}
    mpl.rc('font', **font)
    
    
    # Plot results 
    plt.plot(X[0,:],X[1,:],'.', color='coral', markersize=marker_size, alpha=0.8)
    plt.plot(X[0,:],X[1,:], color='blue', linewidth=line_width, alpha=0.6)
    plt.plot(X[0,0],X[1,0],'kx')
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    
elif f_plot_option == 1 :
    # Style plot 
    marker_size = 1.5
    line_width = 1.25
    fig = plt.figure(figsize=(20,5))
    axis_font = 15
    ax_label_font = 15
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : axis_font}
    mpl.rc('font', **font)


    # Plot results 
    X = desired_trajectory
    ax1 = fig.add_subplot(121)
    ax1.grid()
    ax1.plot(X[0,:],X[1,:],'.', color='coral', markersize=marker_size, alpha=0.8)
    ax1.plot(X[0,:],X[1,:], color='blue', linewidth=line_width, alpha=0.6)
    ax1.plot(X[0,0],X[1,0],'kx')
    ax1.set_xlabel("$x-position$", fontsize=ax_label_font)
    ax1.set_ylabel("$y-position$", fontsize=ax_label_font)
    # ax1.xlim([-10, 10])
    # ax1.ylim([-10, 10])
    
    ax2 = fig.add_subplot(122)
    ax2.grid()
    ax2.plot(time_vector, X[2,:],'.', color='r', markersize=marker_size, alpha=0.2)
    ax2.plot(time_vector, X[3,:],'.', color='b', markersize=marker_size, alpha=0.2)
    ax2.plot(time_vector, X[2,:], color='red', linewidth=line_width, alpha=0.6)
    ax2.plot(time_vector, X[3,:], color='blue', linewidth=line_width, alpha=0.6)
    ax2.set_xlabel("time", fontsize=ax_label_font)
    ax2.set_ylabel("velocity", fontsize=ax_label_font)
    
elif f_plot_option == 2 :
    # Style plot 
    marker_size = 1.5
    line_width = 1.25
    fig = plt.figure(figsize=(20,5))
    axis_font = 9
    ax_label_font = 11
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : axis_font}
    mpl.rc('font', **font)


    # Plot results 
    X = desired_trajectory
    ax1 = fig.add_subplot(131)
    ax1.grid()
    ax1.plot(X[0,:],X[1,:],'.', color='coral', markersize=marker_size, alpha=0.8)
    ax1.plot(X[0,:],X[1,:], color='blue', linewidth=line_width, alpha=0.6)
    ax1.plot(X[0,0],X[1,0],'kx')
    ax1.set_xlabel("x-position", fontsize=ax_label_font)
    ax1.set_ylabel("y-position", fontsize=ax_label_font)
    # ax1.xlim([-10, 10])
    # ax1.ylim([-10, 10])
    
    ax2 = fig.add_subplot(132)
    ax2.grid()
    ax2.plot(time_vector, X[2,:],'.', color='r', markersize=marker_size, alpha=0.2)
    ax2.plot(time_vector, X[3,:],'.', color='b', markersize=marker_size, alpha=0.2)
    ax2.plot(time_vector, X[2,:], color='red', linewidth=line_width, alpha=0.6)
    ax2.plot(time_vector, X[3,:], color='blue', linewidth=line_width, alpha=0.6)
    ax2.set_xlabel("time$", fontsize=ax_label_font)
    ax2.set_ylabel("velocity", fontsize=ax_label_font)
    
    ax3 = fig.add_subplot(133)
    ax3.plot(time_vector, U[0,:], '.', color='red', markersize=marker_size, alpha=0.6)
    ax3.plot(time_vector, U[1,:], '.', color='blue', markersize=marker_size, alpha=0.6)
    ax3.grid()
    ax3.set_xlabel("time", fontsize=ax_label_font)
    ax3.set_ylabel("thrust force", fontsize=ax_label_font)

# End 
print("complete")





