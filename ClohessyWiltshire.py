#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class for Clohessy Wiltshire Dynamics

Author: Christopher W. Hays
"""
from parameters import SystemParameters
import numpy as np


class ClohessyWiltshire():

    #Pull parameters
    sys_data = SystemParameters() 

    # Assign Parameters
    mean_motion = sys_data.mean_motion
    mass_chaser = sys_data.mass_chaser

    # Define CWH Dynamics 
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
    
    @classmethod
    def CW(cls, state_vector=np.zeros([6,1], dtype=float), control_vector=np.zeros([6,1], dtype=float)):
        return np.matmul(cls.A,state_vector)+np.matmul(cls.B, control_vector)