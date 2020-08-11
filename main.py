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
from ClohessyWiltshire import ClohessyWiltshire
from mpl_toolkits.mplot3d import Axes3D
from utilities.misc import probViolation


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Import the Desired Controller from the "controllers" directory 
from controllers.template_controller import Controller
# Import the Desired Measurement Model 
from MeasurementModels.angles_only import MeasurementModel
# Import the Desired Filter from the "filters" directory 
from filters.ExtendedKalmanFilter import dynamicFilter
# Import Active Set Invariance Filter (ASIF) (aka RTA mechanism)
# from asif.CBF_for_speed_limit import ASIF
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


##############################################################################
#                                 Set up                                     #
##############################################################################

# Flags 
f_plot_option = 4 # choose 0, 1, 2, or 3 
f_save_plot = True # saves a plot at end of simulation 
f_use_RTA_filter = False # filters the controllers input to ensure safety 
adaptiveMeasurement = False # True or False; False sets to default measurement frequency

# Parameters 
T  = 5000 # total simulation time [s]
Nsteps = T # number steps in simulation time horizon 

dim_state = 6; 
dim_control = 3; 
dim_meas = MeasurementModel.dimension
sys_data = SystemParameters() 
mean_motion = sys_data.mean_motion
mass_chaser = sys_data.mass_chaser # [kg]
Fmax = sys_data.max_available_thrust # [N]
T_sample = sys_data.controller_sample_period # [s]
T_meas = sys_data.filter_sample_period # [s]

# Initial Values
x = 10 # [m]
x_dot = 0 
x0 = np.array([[x],  # x
               [2/mean_motion*x_dot],  # y 
               [0],  # z
               [0],  # xdot 
               [-2*mean_motion*x],  # ydot 
               [0]]) # zdot 
u0 = np.array([[0],  # Fx 
               [0],  # Fy
               [0]]) # Fz

# Setup filter paramters
x_hat = x0 + np.array([[rand.random()],
                    [rand.random()],
                    [0],
                    [0],
                    [0],
                    [0]])

P0 = np.identity(6)*0.1


##############################################################################
#                                Simulate                                    #
##############################################################################

# Set up simulation 
t = np.linspace(0, T, Nsteps) # evenly spaced time instances in simulation horizon 
X = np.zeros([dim_state, Nsteps]) # state at each time 
U = np.zeros([dim_control, Nsteps]) # control at each time 
X_hat = np.zeros([dim_state, Nsteps]) # state estimate at each time step
state_error = np.zeros([dim_state, Nsteps]) # State error at each time step
P = np.zeros([dim_state, dim_state, Nsteps]) # Covariance Matrix
X_meas = np.zeros([dim_meas, Nsteps])
dt = t[1]-t[0]
X[:,0]=x0.reshape(dim_state)
X_hat[:,0] = x_hat.reshape(dim_state)
state_error[:,0] = X_hat[:,0]-X[:,0] # state error at initial time step
P[:,:,0] = P0 # Covariance at initial time step
controller = Controller() # Initialize Controller class 
#asif = ASIF() # Initialize ASIF class 
filterScheme = dynamicFilter() # Initialize filter class
takeMeasurement = MeasurementModel() # Define Measurement Model
X_meas[:,0] = takeMeasurement.h(X[:,0]).reshape(1,dim_meas)
violation_probability = np.zeros([Nsteps-1])

steps_per_sample = np.max([1, np.round(T_sample/dt)])
effective_controller_period = steps_per_sample*dt 

steps_per_meas = np.max([1, np.round(T_meas/dt)])
measurement_period = steps_per_meas*dt

# Setup Safety Distances
delta_min = 1
delta_max = 20

print("\nSimulating with time resolution "+"{:.2f}".format(dt)+
      " s and controller period "+"{:.2f}".format(effective_controller_period)+" s"+ 
      " and measurement period "+"{:.2f}".format(measurement_period)+" s \n")

# Iterate over time horizon 
for i in range(1,Nsteps):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Call Controller
    if (i-1)%steps_per_sample == 0: 
        u = controller.main(X_hat[:,i-1], (i-1)*dt)  

    # Filter Input
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

    # Compute Violation Probability====================================================
    # Only measure if uncertainty is high enough
    violation_probability[i-1] = probViolation(X_hat[:,i-1], P[:,:,i-1], delta_min, delta_max)
    #==================================================================================

    # Compute Measurement
    if adaptiveMeasurement:
        if violation_probability[i-1] <= 0.95:
            #x_meas = MeasurementModel.MeasureFcn(X[:,i])
            x_meas = MeasurementModel.h(X[:,i])
            X_meas[:,i] = x_meas.reshape(1,dim_meas)
            x_meas = x_meas
            #print("Measurement taken at index"+"{:f}".format(i-1))
        else:
            x_meas = 'NA'
    else:
        if (i-1)%steps_per_meas==0:
            x_meas = MeasurementModel.h(X[:,i])
            X_meas[:,i] = x_meas.reshape(1,dim_meas)
            x_meas = x_meas
            #print("Measurement taken at index"+"{:f}".format(i-1))          
        else:
            x_meas = 'NA'        


    # Run Filter
    x_hat, P[:,:,i] = filterScheme.main(x_hat, x_meas, P[:,:,i-1], u, dt, MeasurementModel)
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
    plt.xlim([-10000, 10000])
    plt.ylim([-10000, 10000])
    
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
    try:
        f_speed_limit_const = asif.safety_constraint # 0 for none, 1 right cone, or 2 for other cone  
    except: 
        f_speed_limit_const = 0 
    if not f_use_RTA_filter:
        f_speed_limit_const = 0 
    
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

    # Plot results 
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.grid()
    ax1.plot(X[0,:],X[1,:],X[2,:],'.', color='coral', markersize=marker_size, alpha=0.8)
    ax1.plot(X[0,:],X[1,:],X[2,:], color='blue', linewidth=line_width, alpha=0.6)
    K = 8000
    ax1.set_xlim( [-K, K] )
    ax1.set_ylim( [-K, K] )
    ax1.set_zlim( [-K, K] )
    # ax1.plot(X[0,0],X[1,0],X[2,0])
    # ax1.plot(0,0,0,'go', alpha=0.5)
    ax1.set_xlabel("x-position", fontsize=ax_label_font)
    ax1.set_ylabel("y-position", fontsize=ax_label_font)
    ax1.set_zlabel("z-position")
    plt.title("Trajectory", fontsize=ax_label_font)

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
    
    ax4 = fig.add_subplot(222, projection='3d')
    plt.title("Position vs Speed")
    ax4.grid()
    vmag = (X[3,:]**2 + X[4,:]**2)**(0.5)
    ax4.plot( X[0,:], X[1,:], vmag, 'r' )
    K2 = 5000
    if f_speed_limit_const >= 0.5:
        x = np.arange(-K2, K2, 10)
        y = np.arange(-K2, K2, 10)
        x, y = np.meshgrid(x, y)
        R = np.sqrt(asif.K)*np.sqrt(x**2 + y**2)
        z = R
    ax4.set_xlim( [-K, K] )
    ax4.set_ylim( [-K, K] )
    ax4.set_zlim( [0, 15] )
    ax4.set_xlabel("x-position", fontsize=ax_label_font)
    ax4.set_ylabel("y-position", fontsize=ax_label_font)
    ax4.set_zlabel("velocity magnitude", fontsize=ax_label_font)

    # Plot the surface.
    if f_speed_limit_const == 1: 
        surf = ax4.plot_surface(x,y,z, cmap=mpl.cm.coolwarm, linewidth=0, antialiased=False, alpha=.3)
    elif f_speed_limit_const == 2: 
        surf = ax4.plot_surface(x,y,np.sqrt(z), cmap=mpl.cm.coolwarm, linewidth=0, antialiased=False, alpha=.3)

elif f_plot_option == 3 :
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
    an = np.linspace(0, 2*np.pi, len(t)) # Parametrize angles for plotting circle

    # Plot results 
    ax1 = fig.add_subplot(131)
    ax1.grid()
    ax1.plot(t, X[0,:], color='red', linewidth=line_width, alpha=0.8,label='Truth')
    ax1.plot(t, X_hat[0,:],color='blue', linewidth=line_width, alpha=0.8,label='Estimated')
    ax1.plot(t, X_hat[0,:]+np.sqrt(P[0,0,:]),'--', color='black',alpha=0.4, label='1-$\sigma$  bounds')
    ax1.plot(t, X_hat[0,:]-np.sqrt(P[0,0,:]),'--', color='black',alpha=0.4)
    #ax1.plot(t, X_meas[0,:],'.',color='green', linewidth=line_width, alpha=0.8,label='Measured')
    #ax1.set_ylim(min(X_hat[0,:]), max(X_hat[0,:]))
    ax1.set_xlabel("$Time (s)$", fontsize=ax_label_font)
    ax1.set_ylabel("$X-Position (m)$", fontsize=ax_label_font)
    plt.title("X-Position vs. Time", fontsize=ax_label_font)
    ax1.legend()

    ax2 = fig.add_subplot(132)
    ax2.grid()
    ax2.plot(t, X[1,:], color='red', linewidth=line_width, alpha=0.8,label='Truth')
    ax2.plot(t, X_hat[1,:],color='blue', linewidth=line_width, alpha=0.8,label='Estimated')
    ax2.plot(t, X_hat[1,:]+np.sqrt(P[1,1,:]),'--', color='black',alpha=0.4, label='1-$\sigma$ bounds')
    ax2.plot(t, X_hat[1,:]-np.sqrt(P[1,1,:]),'--', color='black',alpha=0.4)
    #ax2.plot(t, X_meas[1,:],'.',color='green', linewidth=line_width, alpha=0.8,label='Measured')
    #ax2.set_ylim(min(X_hat[1,:]),max(X_hat[1,:]))
    ax2.set_xlabel("$Time (s)$", fontsize=ax_label_font)
    ax2.set_ylabel("$Y-Position (m)$", fontsize=ax_label_font)
    plt.title("Y-Position vs. Time", fontsize=ax_label_font)
    ax2.legend()


    ax3 = fig.add_subplot(133)
    ax3.plot(t, state_error[0,:], color='blue', markersize=marker_size, alpha=0.8,label='x-error')
    ax3.plot(t, state_error[1,:], color='red', markersize=marker_size, alpha=0.8, label='y-error')
    ax3.grid()
    ax3.set_xlabel("Time", fontsize=ax_label_font)
    ax3.set_ylabel("State Error", fontsize=ax_label_font)
    #ax3.set_ylim(min(state_error[0,1:]), max(state_error[1,1:]))
    plt.title("State Error vs. Time", fontsize=ax_label_font)
    ax3.legend()


    # Save and Show 
    if f_save_plot: 
        plt.savefig('estimation_plot')
        plt.show()

elif f_plot_option == 4 :

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
    an = np.linspace(0, 2*np.pi, 100) # Parametrize angles for plotting circle

    # Plot results 
    ax1 = fig.add_subplot(121)
    ax1.grid()
    ax1.plot(X[0,:], X[1,:], color='red', linewidth=line_width, alpha=0.8,label='Truth')
    ax1.plot(X_hat[0,:], X_hat[1,:], color='blue', linewidth=line_width, alpha=0.8,label='Estimated')
    ax1.plot(X_hat[0,:]+np.sqrt(P[0,0,:]), X_hat[1,:]+np.sqrt(P[1,1,:]),'--',color='k', linewidth=line_width, alpha=0.8,label='1-$\sigma$ bounds')
    ax1.plot(X_hat[0,:]-np.sqrt(P[0,0,:]), X_hat[1,:]-np.sqrt(P[1,1,:]),'--',color='k', linewidth=line_width, alpha=0.8)
    #ax1.set_aspect('equal', 'box')
    ax1.plot(delta_min*np.cos(an), delta_min*np.sin(an),linewidth=line_width, color='coral',label='Min Range')
    ax1.plot(delta_max*np.cos(an), delta_max*np.sin(an),linewidth=line_width, color='green',label='Max Range')
    ax1.legend()
    ax1.set_ylabel('Y (m)')
    ax1.set_xlabel('X (m)')
    plt.title('Chaser Position')


    ax2 = fig.add_subplot(122)
    ax2.grid()
    ax2.plot(t[1:], violation_probability, color='red', linewidth=1.5*line_width)
    ax2.plot(t[0:], 0.95*np.ones([len(t)]), '--', color='black', linewidth=line_width, label='Threshold')
    ax2.set_ylabel('Probability In Safe Zone')
    ax2.set_xlabel('Time (s)')
    plt.title('Probability Chaser is in Safe Zone')
    ax2.set_xlim([0, t[-1]])

# Save and Show 
    if f_save_plot: 
        plt.savefig('estimation_plot')
        plt.show()


# End 
print("complete")





