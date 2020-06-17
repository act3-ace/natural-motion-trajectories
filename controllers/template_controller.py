#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: <your name here>

A template for creating controllers, returns zero input 

The main loop will call the controller's "main" class and expect 
a control signal of appropriate size and type to be returned 

In order for this script to work correctly with the main script, avoid 
changing the name of the "Controller" class, or the inputs and outputs of 
the "main" function.  

"""

import numpy as np 
import os, sys, inspect
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))) # add parent directory to path  
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utilities')) # add utilities to path 
from parameters import SystemParameters 


class Controller(SystemParameters): 
    def __init__(self):
        self.zero_input = np.zeros([3,1])
        
        ######################################################################
        # Set up the controller parameters and options here
        ######################################################################
        
        
    def main(self, x0, t):
        """
        Parameters
        ----------
        x : numpy array with 6 elements 
            x = [x_pos, y_pos, z_pos, x_vel, y_vel, z_vel]
            where x,y,z are hill frame coordinates 

        Returns
        -------
        u : 3x1 numpy array
            u = [[Fx], [Fy], [Fz]]
            elements represent forces along x, y, and z axes respectively 
        """
        
        ######################################################################
        # Insert your code here 
        ######################################################################
        
        u = self.zero_input
        
        return u
        
        
        