#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: <your name here>

Chris: Add in message on how to use filter class...

"""

import numpy as np 
import os, sys, inspect
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))) # add parent directory to path  
from parameters import SystemParameters 


class Filter(SystemParameters): 
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
        
        x_hat = self.zero_input
        
        return x_hat
        
        
        