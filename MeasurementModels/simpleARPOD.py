#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


"""

import numpy as np 
import random 

class MeasurementModel():
 
    Q = np.zeros(6) # Process Noise
        
    std_dev = 0.0001
    R = np.identity(6)*std_dev

    @classmethod
    def MeasureFcn(cls, state_real):
        return np.random.multivariate_normal(state_real, cls.R)

    @classmethod
    def BuildMeasureJacob(cls,state_real):
        rho = np.linalg.norm(state_real[1:3])

        cls.H[1,1] = state_real[1]/rho
        cls.H[1,2] = state_real[2]/rho
        cls.H[1,3] = 0
        cls.H[2,1] =  -state_real[2]/rho
        cls.H[2,2] = state_real[1]/rho
        cls.H[2,3] = 0
        cls.H[3,1] = 0
        cls.H[3,2] = 0
        cls.H[3,3] = 1/rho

        return cls.H # Measurement Model Jacobian