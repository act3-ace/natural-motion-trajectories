#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


"""

import numpy as np 
import random 

class MeasurementModel():
 
    Q = np.zeros([6,6]) # Process Noise
        
    std_dev = 0.01
    R = np.identity(3)*std_dev

    H = np.zeros([3,6])

    dimension = 3

    @classmethod
    def MeasureFcn(cls, state_real):
        return np.random.multivariate_normal(state_real, cls.R)

    @classmethod
    def h(cls, state_real):

        vec_size = np.shape(state_real)

        # Ensure column vector
        if vec_size[0] == 6:
            state_real = np.reshape(state_real, [6,1])
        
        # Initialize measurement vector
        s = np.zeros([3,1])
        
        #s = [r/||r||; v]
        rho = np.linalg.norm(state_real[0:3])
        s[0:3] = state_real[0:3]/rho
        #s[3:6] = state_real[3:6]
        
        return s

    @classmethod
    def BuildMeasureJacob(cls,state_real):
        rho = np.linalg.norm(state_real[0:3])
        s = state_real[0:3]/rho

        cls.H[0:3,0:3] = 1/rho*(np.identity(3)-s*np.transpose(s))

        return cls.H # Measurement Model Jacobian