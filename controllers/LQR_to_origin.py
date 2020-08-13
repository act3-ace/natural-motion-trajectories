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
        
        # Assign Parameters
        mean_motion = self.mean_motion
        mass_chaser = self.mass_chaser
    
        # Define in-plane CWH Dynamics 
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
        
        # Specify LQR gains 
        Q = np.multiply(.050,np.eye(6))   # State cost 
        R = np.multiply(1000,np.eye(3))   # Control cost 
        
        # Calculate LQR Cost 
        self.Klqr = -misc.get_lqr_gain(A, B, Q, R)
                
        
    def main(self, x0, t):
        xd = np.array([0, 0, 0, 0, 0, 0])
        # print(x0)
        # print(xd)
        return np.matmul(self.Klqr,x0-xd).reshape(3,1)
        
    
    
    
    
        
        