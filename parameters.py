#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class specifies all system parameters 

"""


class SystemParameters:
    # Physical  
    mean_motion = 0.001027; # [rad/s]   
    mass_chaser = 140 # [kg]
    
    max_available_thrust = 20 # [N] 
    
    controller_sample_rate = 10 # [Hz] 
    controller_sample_period = 1/controller_sample_rate # [s]
    
    

    
    