#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mark's version of main script for running simulation 

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
from ClohessyWiltshire import ClohessyWiltshire
from mpl_toolkits.mplot3d import Axes3D


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Import the Desired Controller from the "controllers" directory 
from controllers.LQR_to_origin import Controller
# Import the Desired Measurement Model 
from MeasurementModels.simple import MeasurementModel
# Import the Desired Filter from the "filters" directory 
from filters.ExtendedKalmanFilter import dynamicFilter
# Import Active Set Invariance Filter (ASIF) (aka RTA mechanism)
from asif.CBF_for_speed_limit import ASIF
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


##############################################################################
#                                 Set up                                     #
##############################################################################

# Flags 
f_plot_option = 3 # choose 0, 1, 2, or 3 
f_save_plot = False # saves a plot at end of simulation 
f_use_RTA_filter = True # filters the controllers input to ensure safety 
f_plot_safety_limit = True # Plots safety constraint given by "v < 2*n*r + 0.2"

# Modifiable Simulation Parameters 
T  = 2000 # total simulation time [s]
Nsteps = int(T) # number steps in simulation time horizon 
assumed_dist_range = 2000 # assumed distance range - affects plots and state initialization

# Other Parameters (see parameters.py for system parameters)
dim_state = 6; 
dim_control = 3; 
sys_data = SystemParameters() 
mean_motion = sys_data.mean_motion
mass_chaser = sys_data.mass_chaser # [kg]
Fmax = sys_data.max_available_thrust # [N]
T_sample = sys_data.controller_sample_period # [s]

# Initial Values
srange = 1*assumed_dist_range # [m]
sx = srange*(np.random.rand()-0.5)
sy = srange*(np.random.rand()-0.5)
vrange = 1.2*mean_motion*np.sqrt(sx**2 + sy**2) #  np.max([2*mean_motion*assumed_dist_range,10])   # [m/s]
# x0 = np.array([ [500], [0], [0], [0], [0], [0]  ] )
x0 = np.array([ [sx],  # x
                [sy],  # y 
                [0],  # z
                [vrange*2*(np.random.rand()-0.5)],  # xdot 
                [vrange*2*(np.random.rand()-0.5)],  # ydot 
                [0]]) # zdot 
u0 = np.array([[0],  # Fx 
               [0],  # Fy
               [0]]) # Fz

# Filter paramters
x_hat = x0

P = np.identity(6)

##############################################################################
#                                Simulate                                    #
##############################################################################

# Set up simulation 
t = np.linspace(0, T, Nsteps) # evenly spaced time instances in simulation horizon 
X = np.zeros([dim_state, Nsteps]) # state at each time 
U = np.zeros([dim_control, Nsteps]) # control at each time 
X_hat = np.zeros([dim_state, Nsteps]) # state estimate at each time step
state_error = np.zeros([dim_state, Nsteps]) # State error at each time step
X_meas = np.zeros([dim_state, Nsteps])
dt = t[1]-t[0]
X[:,0]=x0.reshape(dim_state)
X_hat[:,0] = x_hat.reshape(dim_state)
state_error[:,0] = X_hat[:,0]-X[:,0] # state error at initial time step
controller = Controller() # Initialize Controller class 
asif = ASIF() # Initialize ASIF class 
filterScheme = dynamicFilter() # Initialize filter class
takeMeasurement = MeasurementModel() # Define Measurement Model
X_meas[:,0] = takeMeasurement.MeasureFcn(X[:,0])


steps_per_sample = np.max([1, np.round(T_sample/dt)])
effective_controller_period = steps_per_sample*dt 
print("\nSimulating with time resolution "+"{:.2f}".format(dt)+
      " s and controller period "+"{:.2f}".format(effective_controller_period)+" s \n")

# Iterate over time horizon 
for i in range(1,Nsteps):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Call Controller
    if (i-1)%steps_per_sample == 0: 
        u = controller.main(X_hat[:,i-1], (i-1)*dt)
        # print("ud = ", u)

        # Filter Input with RTA 
        if f_use_RTA_filter: 
            u = asif.main(X_hat[:,i-1], u)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    # Saturate 
    for j in range(3):
        if u[j,0] > Fmax: 
            u[j,0] = Fmax 
        elif u[j,0] < -Fmax: 
            u[j,0] = -Fmax 
        
    U[:,i] = u.reshape(dim_control) # record history of control inputs (optional)
    
    # Propagate 
    xdot = ClohessyWiltshire.CW(X[:,i-1].reshape(dim_state,1) , u)*dt
    X[:,i] = X[:,i-1] + xdot.reshape(dim_state)

    # Compute Measurement
    x_meas = MeasurementModel.MeasureFcn(X[:,i])
    X_meas[:,i] = x_meas
    x_meas = x_meas.reshape(dim_state,1)

    # Run Filter
    x_hat, P = filterScheme.main(x_hat, x_meas, P, u, dt, MeasurementModel)
    X_hat[:,i] = x_hat.reshape(dim_state)

    # Calculate state error
    state_error[:,i] = X_hat[:,i]-X[:,i]

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
    plt.xlabel("x-position ($s_x$) [m]", fontsize=ax_label_font)
    plt.ylabel("y-position ($s_y$) [m]", fontsize=ax_label_font)
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : axis_font}
    mpl.rc('font', **font)
    
    # Plot results 
    plt.plot(X[0,:],X[1,:],'.', color='coral', markersize=marker_size, alpha=0.8)
    plt.plot(X[0,:],X[1,:], color='blue', linewidth=line_width, alpha=0.6)
    plt.plot(X[0,0],X[1,0],'kx')
    plt.xlim([-assumed_dist_range, assumed_dist_range])
    plt.ylim([-assumed_dist_range, assumed_dist_range])
    
elif f_plot_option == 1 :
    # Style plot 
    marker_size = 1.5
    line_width = 1.25
    fig = plt.figure(figsize=(15,6))
    axis_font = 15
    ax_label_font = 15
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : axis_font}
    mpl.rc('font', **font)

    # Plot results 
    ax1 = fig.add_subplot(121)
    ax1.grid()
    ax1.plot(X[0,:]*0.001,X[1,:]*0.001, '.', color='coral', markersize=marker_size, alpha=0.8)
    ax1.plot(X[0,:]*0.001,X[1,:]*0.001, color='blue', linewidth=line_width, alpha=0.6)
    ax1.plot(X[0,0],X[1,0],'kx')
    plt.xlabel("x-position ($s_x$) [km]", fontsize=ax_label_font)
    plt.ylabel("y-position ($s_y$) [km]", fontsize=ax_label_font)
    plt.xlim([-assumed_dist_range*0.001, assumed_dist_range*0.001])
    plt.ylim([-assumed_dist_range*0.001, assumed_dist_range*0.001])
    
    ax2 = fig.add_subplot(122)
    ax2.grid()
    ax2.plot(t, X[3,:],'.', color='r', markersize=marker_size, alpha=0.5)
    ax2.plot(t, X[4,:],'.', color='b', markersize=marker_size, alpha=0.5)
    ax2.plot(t, X[3,:], color='red', linewidth=line_width, alpha=0.2)
    ax2.plot(t, X[4,:], color='blue', linewidth=line_width, alpha=0.2)
    ax2.set_xlabel("time ($t$) [s]", fontsize=ax_label_font)
    ax2.set_ylabel("velocity components ($v_x,v_y$) [m/s]", fontsize=ax_label_font)
    
elif f_plot_option == 2 :

    # Style plot 
    marker_size = 1.5
    line_width = 1.25
    fig = plt.figure(figsize=(10,10))
    axis_font = 9
    ax_label_font = 11
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : axis_font}
    mpl.rc('font', **font)

    # Plot 1 - Trajectory in 3d space 
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.grid()
    ax1.plot(X[0,:]*0.001,X[1,:]*0.001,X[2,:]*0.001,'.', color='coral', markersize=marker_size, alpha=0.8)
    ax1.plot(X[0,:]*0.001,X[1,:]*0.001,X[2,:]*0.001, color='blue', linewidth=line_width, alpha=0.6)
    K = assumed_dist_range*0.001
    ax1.set_xlim( [-K, K] )
    ax1.set_ylim( [-K, K] )
    ax1.set_zlim( [-K, K] )
    ax1.set_xlabel("x-position ($s_x$) [km]", fontsize=ax_label_font)
    ax1.set_ylabel("y-position ($s_y$) [km]", fontsize=ax_label_font)
    ax1.set_zlabel("z-position ($s_z$) [km]", fontsize=ax_label_font)
    plt.title("Trajectory", fontsize=ax_label_font)
    
    # Plot 2 - Position vs speed 
    ax4 = fig.add_subplot(222, projection='3d')
    plt.title("Position vs Speed")
    ax4.grid()
    vmag = (X[3,:]**2 + X[4,:]**2)**(0.5)
    ax4.plot( X[0,:]*0.001, X[1,:]*0.001, vmag, 'r' )

    ax4.set_xlim( [-K, K] )
    ax4.set_ylim( [-K, K] )
    ax4.set_zlim( [0, 2*vrange] )
    ax4.set_xlabel("x-position ($s_x$) [km]", fontsize=ax_label_font)
    ax4.set_ylabel("y-position ($s_x$) [km]", fontsize=ax_label_font)
    ax4.set_zlabel("speed ($\Vert v \Vert_2$) [m/s]", fontsize=ax_label_font)
    
    if f_plot_safety_limit:
        x = np.arange(-K*1000, K*1000, 100)
        y = np.arange(-K*1000, K*1000, 100)
        x, y = np.meshgrid(x, y)
        R = 2*mean_motion*np.sqrt((x)**2 + (y)**2)
        z = R + 0.2
        surf = ax4.plot_surface(x*0.001,y*0.001,z, cmap=mpl.cm.coolwarm, linewidth=0, antialiased=False, alpha=.25)

    # Plot 3 - Thrust vs Time 
    ax3 = fig.add_subplot(223)
    ax3.plot(t, U[0,:], '.', color='red', markersize=marker_size, alpha=0.8)
    ax3.plot(t, U[1,:], '.', color='blue', markersize=marker_size, alpha=0.8)
    ax3.plot(t, U[2,:], '.', color='green', markersize=marker_size, alpha=0.8)
    ax3.plot(t, U[0,:], color='red', linewidth=line_width, alpha=0.2)
    ax3.plot(t, U[1,:], color='blue', linewidth=line_width, alpha=0.2)
    ax3.plot(t, U[2,:], color='green', linewidth=line_width, alpha=0.2)
    ax3.grid()
    ax3.set_xlabel("time", fontsize=ax_label_font)
    ax3.set_ylabel("thrust force", fontsize=ax_label_font)
    plt.title("Thrust vs. Time", fontsize=ax_label_font)

    # Plot 4 - Velocity vs Time 
    ax2 = fig.add_subplot(224)
    ax2.grid()
    ax2.plot(t, X[3,:],'.', color='r', markersize=marker_size, alpha=0.2)
    ax2.plot(t, X[4,:],'.', color='b', markersize=marker_size, alpha=0.2)
    ax2.plot(t, X[5,:],'.', color='g', markersize=marker_size, alpha=0.2)
    ax2.plot(t, X[3,:], color='red', linewidth=line_width, alpha=0.6)
    ax2.plot(t, X[4,:], color='blue', linewidth=line_width, alpha=0.6)
    ax2.plot(t, X[5,:], color='green', linewidth=line_width, alpha=0.6)
    ax2.set_xlabel("time", fontsize=ax_label_font)
    ax2.set_ylabel("velocity", fontsize=ax_label_font)
    plt.title("Velocity vs. Time", fontsize=ax_label_font)


    

elif f_plot_option == 3 :
    # Style plot 
    marker_size = 1.5
    line_width = 1.25
    fig = plt.figure(figsize=(10,10))
    axis_font = 9
    ax_label_font = 11
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : axis_font}
    mpl.rc('font', **font)

    # Plot 1 - Trajectory in 3d space 
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.grid()
    ax1.plot(X[0,:]*0.001,X[1,:]*0.001,X[2,:]*0.001,'.', color='coral', markersize=marker_size, alpha=0.8)
    ax1.plot(X[0,:]*0.001,X[1,:]*0.001,X[2,:]*0.001, color='blue', linewidth=line_width, alpha=0.6)
    K = assumed_dist_range*0.001
    ax1.set_xlim( [-K, K] )
    ax1.set_ylim( [-K, K] )
    ax1.set_zlim( [-K, K] )
    ax1.set_xlabel("x-position ($s_x$) [km]", fontsize=ax_label_font)
    ax1.set_ylabel("y-position ($s_y$) [km]", fontsize=ax_label_font)
    ax1.set_zlabel("z-position ($s_z$) [km]", fontsize=ax_label_font)
    plt.title("Trajectory", fontsize=ax_label_font)
    
    # Plot 2 - Position vs speed 
    ax4 = fig.add_subplot(222, projection='3d')
    plt.title("Position vs Speed")
    ax4.grid()
    vmag = (X[3,:]**2 + X[4,:]**2)**(0.5)
    ax4.plot( X[0,:]*0.001, X[1,:]*0.001, vmag, 'r' )

    ax4.set_xlim( [-K, K] )
    ax4.set_ylim( [-K, K] )
    ax4.set_zlim( [0, 2*vrange] )
    ax4.set_xlabel("x-position ($s_x$) [km]", fontsize=ax_label_font)
    ax4.set_ylabel("y-position ($s_x$) [km]", fontsize=ax_label_font)
    ax4.set_zlabel("speed ($\Vert v \Vert_2$) [m/s]", fontsize=ax_label_font)
    
    if f_plot_safety_limit:
        x = np.arange(-K*1000, K*1000, 100)
        y = np.arange(-K*1000, K*1000, 100)
        x, y = np.meshgrid(x, y)
        R = 2*mean_motion*np.sqrt((x)**2 + (y)**2)
        z = R + 0.2
        surf = ax4.plot_surface(x*0.001,y*0.001,z, cmap=mpl.cm.coolwarm, linewidth=0, antialiased=False, alpha=.25)

    # Plot 3 - Thrust vs Time 
    ax3 = fig.add_subplot(223)
    ax3.plot(t, U[0,:], '.', color='red', markersize=marker_size, alpha=0.8)
    ax3.plot(t, U[1,:], '.', color='blue', markersize=marker_size, alpha=0.8)
    ax3.plot(t, U[2,:], '.', color='green', markersize=marker_size, alpha=0.8)
    ax3.plot(t, U[0,:], color='red', linewidth=line_width, alpha=0.2)
    ax3.plot(t, U[1,:], color='blue', linewidth=line_width, alpha=0.2)
    ax3.plot(t, U[2,:], color='green', linewidth=line_width, alpha=0.2)
    ax3.grid()
    ax3.set_xlabel("time", fontsize=ax_label_font)
    ax3.set_ylabel("thrust force", fontsize=ax_label_font)
    plt.title("Thrust vs. Time", fontsize=ax_label_font)

    # Plot 4 - Distance vs Speed 
    ax2 = fig.add_subplot(224)
    ax2.grid()
    plt.title("Distance vs Speed")
    ax2.set_xlabel("distance from target", fontsize=ax_label_font)
    ax2.set_ylabel("speed", fontsize=ax_label_font)
    # vmag = np.maximum(np.abs(X[3,:]), np.abs(X[4,:]))
    # rmag = np.maximum(np.abs(X[0,:]), np.abs(X[1,:]))
    vmag = np.sqrt( X[3,:]**2 + X[4,:]**2 )
    rmag = np.sqrt( X[0,:]**2 + X[1,:]**2 )
    ax2.plot( rmag, vmag, '.')
    ax2.plot( rmag[0], vmag[0], 'kx')
    ax2.set_ylim( [0, 4 ] )
    ax2.set_xlim( [0, srange] )
    ax2.plot([0,10000], [0, 10000*2*mean_motion], '--' , color='purple', alpha=0.5)
    ax2.plot([0,10000], [0, 10000*0.5*mean_motion], '--', color='purple', alpha=0.5 )
    if f_plot_safety_limit: 
        ax2.plot([0,10000], [asif.K2_s, asif.K2_s + 10000*asif.K1_s], color='r', alpha=0.85)
    

# Save and Show 
if f_save_plot: 
    plt.savefig('trajectory_plot')
    plt.show()

# End 
print("complete")





