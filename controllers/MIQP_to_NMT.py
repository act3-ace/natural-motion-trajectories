#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mark 

A quadratic programming (QP) controller that satisfies the following safety 
constraints: 
    (i) max thrust: Fx, Fy \in [Fmin, Fmax]
    (ii) max velocity: xdot, ydot \in [vmin, vmax]


Assumes: Full 3-D translational dynamics  

The optimization problem is solved at each call of the controller 

Key Paramters:
    self.f_goal_set: specifies the constraint on the final point in the trajectory 
        = 0 to drive the controller to the origin 
        = 1 to drive the controller to a stationary point (along y-axis)
    self.total_plan_time: total time to the goal 
        * note: if this is too short, then no feasible solutions will be found 
          this problem can easily be fixed by adding some number of iterations 
          to the horizon until a solution is found 
    self.tau0: number of steps in the initial planning horizon
    
Note: the time period between steps in the planner "self.dt_plan" 
will remain constant, and the number of steps in the time horizon "tau" will 
be decreased as the vehicle approaches the goal 


TODO: Increase time horizon if first iteration is infeasable 
TODO: Add safety constraint number 3, defining a position dependent speed limit 
TODO: regulation to trajectory point once reached 

"""

import numpy as np 
import os, sys, inspect
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))) 
from parameters import SystemParameters 
import gurobipy as gp 
from gurobipy import GRB


class Controller(SystemParameters): 
    def __init__(self):
        
        # Options 
        self.f_goal_set = 2 # 0 for origin, 1 for periodic line, 2 for ellipses
        self.f_collision_avoidance = True # activates or deactivates collision avoidance requirement
        
        self.total_plan_time = 1000 # time to goal [s] 
        self.tau0 = 50 # number steps in initial planning horizon 
        self.collision_dist = 500 # minimum allowable distance from target [m]
        self.kappa_speed = 2.1*self.mean_motion # NOTE: must be greater than 2*mean_motion
        self.semiminor_out = 5/self.mean_motion # semi-minor axis of outer ellipse bound - motivated by velocity constraint 
        self.semiminor_in = np.sqrt(5/4)*self.collision_dist # semi-minor axis of inner ellipse bound - motivated by collision avoidance constraint 

        
        # Set up (don't modify )
        self.zero_input = np.zeros([3,1])
        self.trajectory_initialized = False 
        if self.f_goal_set == 0:
            self.f_collision_avoidance = False
        
        self.t = np.linspace(0,self.total_plan_time, self.tau0) # time vector 
        self.dt_plan = self.t[1]-self.t[0] # time resolution of solver 
        
        # self.xstar = 0 # optimal trajectory points 
        self.ustar = 0 # optimal control points 
        
        if self.f_goal_set == 0: 
            print("\nDriving chaser to target! \n")
        elif self.f_goal_set == 1: 
            print("\nDriving chaser to stationary trajectory! \n")
        elif self.f_goal_set == 2: 
            print("\nDriving chaser to elliptical NMT! \n")

        
        
    def main(self, x0, t):
        """
        Computes trajectory and uses first input 
        
        """
        
        # Try to find optimal trajectory
        try: 
            self.calculate_trajectory(x0, t) # finds traj starting at x0 
        except: 
            try: 
                self.tau0 = self.tau0 + 20
                self.calculate_trajectory(x0, t) # finds traj starting at x0 
            except:
                print("\nFailed to find trajectory at t = "+str(t),"\n")
        
        u = self.ustar[:,0]
            
        return u.reshape(3,1)
        
    
    def calculate_trajectory(self, x0, t_elapsed):
        """
        Uses Gurobi to calculate optimal trajectory points (self.xstar) and 
        control inputs (self.ustar)

        """
        
        # Options 
        Nin = 6 # number of sides in inner-ellipse polygon approx 
        Nout = 15 # number of sides in outer-ellipse polygon approx 
        
        
        initial_state = x0.reshape(6)
        goal_state = np.zeros(6)
        n  = self.mean_motion
        mc = self.mass_chaser

        # Shorten the number of initial time steps (self.tau0) based on the amount of time elapsed        
        tau = int(max(10, np.round(self.tau0 - t_elapsed/self.dt_plan) ) )
        print("time elapsed = ", t_elapsed )
        
        # Set Ranges 
        smax = 15000 # arbitrary (included bounds to speed up solver)
        vmax = 10 # [m/s] max velocity 
        Fmax = 2 # [N] max force 
        
        
        # Initialize states 
        sx = [] 
        sy = [] 
        sz = [] 
        vx = [] 
        vy = [] 
        vz = [] 
        Fx = [] 
        Fy = [] 
        Fz = [] 
        snorm = [] 
        sxabs = [] 
        syabs = [] 
        vnorm = [] 
        vxabs = [] 
        vyabs = [] 
        zeta = [] 
        
        m = gp.Model("QPTraj")
        
        # Define variables at each of the tau timesteps 
        for t in range(tau) : 
            sx.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -smax, ub = smax, name="sx"+str(t) )) 
            vx.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -vmax, ub = vmax, name="vx"+str(t) )) 
            Fx.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -Fmax, ub = Fmax, name="Fx"+str(t) )) 
            sy.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -smax, ub = smax, name="sy"+str(t) )) 
            vy.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -vmax, ub = vmax, name="vy"+str(t) )) 
            Fy.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -Fmax, ub = Fmax, name="Fy"+str(t) )) 
            sz.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -smax, ub = smax, name="sz"+str(t) )) 
            vz.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -vmax, ub = vmax, name="vz"+str(t) )) 
            Fz.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -Fmax, ub = Fmax, name="Fz"+str(t) ))
            snorm.append( m.addVar(vtype=GRB.CONTINUOUS, lb = 0, ub = smax, name="snorm"+str(t) )) 
            sxabs.append( m.addVar(vtype=GRB.CONTINUOUS, lb = 0, ub = smax, name="sxabs"+str(t) )) 
            syabs.append( m.addVar(vtype=GRB.CONTINUOUS, lb = 0, ub = smax, name="syabs"+str(t) )) 
            vnorm.append( m.addVar(vtype=GRB.CONTINUOUS, lb = 0, ub = vmax, name="vnorm"+str(t) )) 
            vxabs.append( m.addVar(vtype=GRB.CONTINUOUS, lb = 0, ub = vmax, name="vxabs"+str(t) )) 
            vyabs.append( m.addVar(vtype=GRB.CONTINUOUS, lb = 0, ub = vmax, name="vyabs"+str(t) )) 
        
        for p in range(Nin): 
            zeta.append( m.addVar(vtype=GRB.BINARY, name="zeta"+str(p)) )
            
        m.update()
                
        # Set Initial Conditions 
        m.addConstr( sx[0] == initial_state[0] , "sx0")
        m.addConstr( sy[0] == initial_state[1] , "sy0")
        m.addConstr( sz[0] == initial_state[2] , "sz0")
        m.addConstr( vx[0] == initial_state[3] , "vx0")
        m.addConstr( vy[0] == initial_state[4] , "vy0")
        m.addConstr( vz[0] == initial_state[5] , "vz0")
        
        
        # Specify Terminal Set 
        if self.f_goal_set == 0: # origin 
            m.addConstr( sx[-1] == goal_state[0] , "sxf")
            m.addConstr( sy[-1] == goal_state[1] , "syf")
            m.addConstr( sz[-1] == goal_state[2] , "szf")
            m.addConstr( vx[-1] == goal_state[3] , "vxf")
            m.addConstr( vy[-1] == goal_state[4] , "vyf")
            m.addConstr( vz[-1] == goal_state[5] , "vzf")
        elif self.f_goal_set == 1: # stationary point or periodic line 
            m.addConstr( sx[-1] == goal_state[0] , "sxf")
            m.addConstr( vx[-1] == goal_state[3] , "vxf")
            m.addConstr( vy[-1] == goal_state[4] , "vyf")
        elif self.f_goal_set == 2: # ellipse 
            m.addConstr( vy[-1] + 2*n*sx[-1] == 0, "ellipse1" )
            m.addConstr( sy[-1] - (2/n)*vx[-1] == 0, "ellipse2" )
            
        
        # Set dynamic speed limit 
        for t in range(tau):
            # Define the norms: 
            m.addConstr( sxabs[t] == gp.abs_(sx[t]) )
            m.addConstr( syabs[t] == gp.abs_(sy[t]) )
            m.addConstr( snorm[t] == gp.max_(sxabs[t], syabs[t]), "snorm"+str(t) )
            m.addConstr( vxabs[t] == gp.abs_(vx[t]) )
            m.addConstr( vyabs[t] == gp.abs_(vy[t]) )
            m.addConstr( vnorm[t] == gp.max_(vxabs[t], vyabs[t]), "vnorm"+str(t) )
            
            # Speed limit constraint: 
            m.addConstr( vnorm[t] <= self.kappa_speed*snorm[t] )
            
        
        # Collision Avoidance Constraint
        if self.f_collision_avoidance: 
            for t in range(tau): 
                m.addConstr( snorm[t] >= self.collision_dist)
            if initial_state[0]<self.collision_dist or initial_state[1]<self.collision_dist: 
                print("\nERROR: Initial position is too close! Collision constraint violated!\n")
        
        # # Final point within [1km-5km] of target 
        # m.addConstr( snorm[-1] <= 2000 )
        # m.addConstr( snorm[-1] >= 1000 )
            
            
        # Terminal constraint: inner polygonal approx on outer ellipse bound 
        Nout = Nout+1 
        aout = self.semiminor_out
        bout = self.semiminor_out*2 
        theta = np.linspace(0, 2*np.pi, Nout)
        for j in range(0,Nout-1): 
            x0 = aout*np.cos(theta[j])
            y0 = bout*np.sin(theta[j])
            x1 = aout*np.cos(theta[j+1])
            y1 = bout*np.sin(theta[j+1])
            alphax = y0-y1 
            alphay = x1-x0
            gamma  = alphay*y1 + alphax*x1
            m.addConstr( alphax*sx[-1] + alphay*sy[-1] >= gamma , "OPA"+str(j) )
                
        # Terminal constraint: outer polygonal approx on inner ellipse bound 
        if self.f_collision_avoidance : 
            a_in = self.semiminor_in
            b_in = self.semiminor_in*2 
            theta = np.linspace(0, 2*np.pi, Nin+1)  
            big_M = 100000
            for j in range(0,Nin): 
                x0 = a_in*np.cos(theta[j])
                y0 = b_in*np.sin(theta[j])
                c1 = (2*x0/(a_in**2))
                c2 = (2*y0/(b_in**2)) 
                cmag = np.sqrt(c1**2 + c2**2)
                c1 = c1/cmag
                c2 = c2/cmag
                m.addConstr( c1*sx[-1] + c2*sy[-1] - c1*x0 - c2*y0 - big_M*zeta[j]  >= - big_M, "IPA"+str(j) ) 
            m.addConstr( sum( zeta[p] for p in range(Nin) ) >= 0.5 )

            
        # Set Dynamics 
        for t in range(tau-1) :
            # Dynamics 
            m.addConstr( sx[t+1] == sx[t] + vx[t]*self.dt_plan , "Dsx_"+str(t))
            m.addConstr( sy[t+1] == sy[t] + vy[t]*self.dt_plan , "Dsy_"+str(t))
            m.addConstr( sz[t+1] == sz[t] + vz[t]*self.dt_plan , "Dsz_"+str(t))
            m.addConstr( vx[t+1] == vx[t] + sx[t]*3*n**2*self.dt_plan + vy[t]*2*n*self.dt_plan + Fx[t]*(1/mc)*self.dt_plan , "Dvx_"+str(t) )
            m.addConstr( vy[t+1] == vy[t] - vx[t]*2*n*self.dt_plan + Fy[t]*(1/mc)*self.dt_plan , "Dvy_"+str(t) )
            m.addConstr( vz[t+1] == vz[t] + (-n**2)*sz[t]*self.dt_plan     + Fz[t]*(1/mc)*self.dt_plan , "Dvz_"+str(t) )
        
        # Set Objective ( minimize: sum(Fx^2 + Fy^2) )
        obj = Fx[0]*Fx[0] + Fy[0]*Fy[0] + Fz[0]*Fz[0]
        for t in range(0, tau):
            obj = obj + Fx[t]*Fx[t] + Fy[t]*Fy[t] + Fz[t]*Fz[t]
        
        
        m.setObjective(obj, GRB.MINIMIZE)
        m.setParam( 'OutputFlag', False )
        
        # Optimize and report on results 
        m.optimize()
        
        
        # Save desired trajectory 
        self.xstar = np.zeros([6, tau]) 
        self.ustar = np.zeros([3, tau])
        self.snorm = np.zeros([tau])
        self.vnorm = np.zeros([tau])
        
        for t in range(tau): # TODO: find quicker way to do this 
            self.xstar[0,t] = m.getVarByName("sx"+str(t)).x
            self.xstar[1,t] = m.getVarByName("sy"+str(t)).x
            self.xstar[3,t] = m.getVarByName("vx"+str(t)).x
            self.xstar[4,t] = m.getVarByName("vy"+str(t)).x
            self.ustar[0,t] = m.getVarByName("Fx"+str(t)).x
            self.ustar[1,t] = m.getVarByName("Fy"+str(t)).x
            self.ustar[2,t] = m.getVarByName("Fz"+str(t)).x
            self.snorm[t]   = m.getVarByName("snorm"+str(t)).x 
            self.vnorm[t]   = m.getVarByName("vnorm"+str(t)).x         
