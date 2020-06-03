#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main script for running simulation 

state = [x y z xdot ydot zdot] (position and velocity in Hill frame)

X[:,i] accesses the ith state 
U[:,i] accesses the ith control 

control = [Fx Fy Fz] (forces in Hill Frame)

"""

import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import random as rand 
import sys 
from parameters import SystemParameters 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Import the Desired Controller from the "controllers" directory 
from controllers.simple_linear_controller import Controller 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


##############################################################################
#                                 Set up                                     #
##############################################################################

# Flags 
f_plot_option = 2 # choose 0, 1, or 2
f_save_plot = False # saves a plot at end of simulation 


# Parameters 
T  = 60 # total simulation time [s]
Nsteps = 500 # number steps in simulation time horizon 

dim_state = 6; 
dim_control = 3; 
sys_data = SystemParameters() 
mean_motion = sys_data.mean_motion
mass_chaser = sys_data.mass_chaser # [kg]
Fmax = sys_data.max_available_thrust # [N]
T_sample = sys_data.controller_sample_period # [s]


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


##############################################################################
#                                Simulate                                    #
##############################################################################

# Set up simulation 
t = np.linspace(0, T, Nsteps) # evenly spaced time instances in simulation horizon 
X = np.zeros([dim_state, Nsteps]) # state at each time 
U = np.zeros([dim_control, Nsteps]) # control at each time 
dt = t[1]-t[0]
X[:,0]=x0.reshape(dim_state)
controller = Controller() # initialize Controller class 

steps_per_sample = np.max([1, np.round(T_sample/dt)])
effective_controller_period = steps_per_sample*dt 
print("\nSimulating with time resolution "+"{:.2f}".format(dt)+
      " s and controller period "+"{:.2f}".format(effective_controller_period)+" s \n")

# Iterate over time horizon 
for i in range(1,Nsteps):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Call Controller
    if (i-1)%steps_per_sample == 0: 
        u = controller.main(X[:,i-1], (i-1)*dt)  
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    # Saturate 
    for j in range(3):
        if u[j,0] > Fmax: 
            u[j,0] = Fmax 
        elif u[j,0] < -Fmax: 
            u[j,0] = -Fmax 
        
    U[:,i] = u.reshape(dim_control) # record history of control inputs (optional)
    
    # Propagate 
    xdot = F(X[:,i-1].reshape(dim_state,1) , u)*dt
    X[:,i] = X[:,i-1] + xdot.reshape(dim_state)


##############################################################################
#                                Plotting                                    #
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
    ax1 = fig.add_subplot(121)
    ax1.grid()
    ax1.plot(X[0,:],X[1,:],'.', color='coral', markersize=marker_size, alpha=0.8)
    ax1.plot(X[0,:],X[1,:], color='blue', linewidth=line_width, alpha=0.6)
    ax1.plot(X[0,0],X[1,0],'kx')
    ax1.set_xlabel("$x-position$", fontsize=ax_label_font)
    ax1.set_ylabel("$y-position$", fontsize=ax_label_font)
    
    ax2 = fig.add_subplot(122)
    ax2.grid()
    ax2.plot(t, X[3,:],'.', color='r', markersize=marker_size, alpha=0.2)
    ax2.plot(t, X[4,:],'.', color='b', markersize=marker_size, alpha=0.2)
    ax2.plot(t, X[3,:], color='red', linewidth=line_width, alpha=0.6)
    ax2.plot(t, X[4,:], color='blue', linewidth=line_width, alpha=0.6)
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
    ax1 = fig.add_subplot(131)
    ax1.grid()
    ax1.plot(X[0,:],X[1,:],'.', color='coral', markersize=marker_size, alpha=0.8)
    ax1.plot(X[0,:],X[1,:], color='blue', linewidth=line_width, alpha=0.6)
    ax1.plot(X[0,0],X[1,0],'kx')
    ax1.plot(0,0,'go', alpha=0.5)
    ax1.set_xlabel("x-position", fontsize=ax_label_font)
    ax1.set_ylabel("y-position", fontsize=ax_label_font)
    
    ax2 = fig.add_subplot(132)
    ax2.grid()
    ax2.plot(t, X[3,:],'.', color='r', markersize=marker_size, alpha=0.2)
    ax2.plot(t, X[4,:],'.', color='b', markersize=marker_size, alpha=0.2)
    ax2.plot(t, X[3,:], color='red', linewidth=line_width, alpha=0.6)
    ax2.plot(t, X[4,:], color='blue', linewidth=line_width, alpha=0.6)
    ax2.set_xlabel("time", fontsize=ax_label_font)
    ax2.set_ylabel("velocity", fontsize=ax_label_font)
    
    ax3 = fig.add_subplot(133)
    ax3.plot(t, U[0,:], '.', color='red', markersize=marker_size, alpha=0.8)
    ax3.plot(t, U[1,:], '.', color='blue', markersize=marker_size, alpha=0.8)
    ax3.plot(t, U[0,:], color='red', linewidth=line_width, alpha=0.2)
    ax3.plot(t, U[1,:], color='blue', linewidth=line_width, alpha=0.2)
    ax3.grid()
    ax3.set_xlabel("time", fontsize=ax_label_font)
    ax3.set_ylabel("thrust force", fontsize=ax_label_font)


# Save and Show 
if f_save_plot: 
    plt.savefig('trajectory_plot.pdf', dpi=400)
    plt.show()

# End 
print("complete")





