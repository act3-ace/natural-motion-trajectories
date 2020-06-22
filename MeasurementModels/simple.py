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
        H = np.identity(6)

        return H # Measurement Model Jacobian