#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mark 

This controller calculates LQR gain matrix during initialization and uses 
this matrix for regulation in main function 
"""

import numpy as np 
import os, sys, inspect

sys.path.insert(0,os.path.dirname( os.path.dirname( os.path.abspath(inspect.getfile(inspect.currentframe())) ) )) # adds parent directory to path 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utilities'))
from parameters import SystemParameters 
import misc 



class Controller(SystemParameters): 
    def __init__(self):
        self.kappa = 1
        # # Assign Parameters
        # mean_motion = self.mean_motion
        # mass_chaser = self.mass_chaser
    
        # # Define in-plane CWH Dynamics 
        # A = np.array([[0, 1.5*mean_motion],
        #               [0, 0]])
        # B = np.array([[1, 0],
        #               [0, 1]])
        
        # # Specify LQR gains 
        # Q = np.multiply(.010,np.eye(6))   # State cost 
        # R = np.multiply(100000,np.eye(3))   # Control cost 
        
        # # Calculate LQR Cost 
        # self.Klqr = -misc.get_lqr_gain(A, B, Q, R)
                
        
    def main(self, x0, t):
        
        e1 = x0[3]-(self.mean_motion/2)*x0[1]
        e2 = x0[4]+(2*self.mean_motion)*x0[0]
        
        u = np.array([ [-1.5*self.mean_motion*e2 - self.kappa*e1], [-self.kappa*e2], [0] ])
        
        
        return u
        
    
    
    
    
        
        