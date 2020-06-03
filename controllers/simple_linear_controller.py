#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mark 

Simple linear controller with gains taken from MATLAB script

"""

import numpy as np 
import os, sys, inspect
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))) 
from parameters import SystemParameters 


class Controller(SystemParameters): 
    def __init__(self):
        # Set Parameters 
        self.K = -np.array([[420.0004,    0,        0,          490.0000,   0.2876,      0],
                            [0,           52.5000,  0,         -0.2876,     175.0000,    0],
                            [0,           0,        34.9999,    0,          0,           175.0000]])    
        
    def main(self, x0, t):
        return np.matmul(self.K,x0).reshape(3,1)
        
        
        