#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Christopher W. Hays

This code returns the state estimate based on an Extended Kalman Filtering scheme.
Please do not change the class name or any of the function definitions as they will cause errors in the code.

"""

import numpy as np
import scipy.linalg as linalg
import os, sys, inspect
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))) # add parent directory to path  
from parameters import SystemParameters 
from ClohessyWiltshire import ClohessyWiltshire


class dynamicFilter(SystemParameters, ClohessyWiltshire): 
    def __init__(self):
        
        self.Q = [] # Process Noise

        self.H = [] # Jacobian of measurement function
        self.R = [] # Measurement Noise
        

    def main(self, est_state, meas_state, P, u, dt, MeasurementModel):

        def predict(self, state, P, u, dt):
            # Propagate State
            F = linalg.expm((ClohessyWiltshire.A)*dt)
            x_pred = np.matmul(F,state)+np.matmul(ClohessyWiltshire.B, u)*dt

            # Predict Covariance
            P_pred = np.matmul(np.matmul(F,P),F.transpose()) + self.Q

            return x_pred, P_pred

        def update(self, x_pred, P_pred, meas_state):
            # Preliminary Info
            v = meas_state-MeasurementModel.h(x_pred) # Difference actual measured state and predicted measured state
            
            # S = HPH'+R
            S = np.matmul(np.matmul(self.H,P_pred),self.H.transpose())+self.R
            # K = PH'S^(-1)
            K = np.matmul(np.matmul(P_pred, self.H.transpose()),linalg.inv(S)) # Kalman Gain

            # Update State Estimation
            x_hat = x_pred+np.matmul(K,v) 

            # Update Covariance Matrix
            # P = (I-KH)P(I-KH)'+KRK'
            #I_KH = np.identity(6) - np.matmul(K,self.H)
            #P = np.matmul(np.matmul(I_KH, P_pred),I_KH.transpose())+np.matmul(np.matmul(K,self.R),K.transpose())
            
            P = P_pred - np.matmul(K,np.matmul(S,np.transpose(K)))
        
            return x_hat, P
        
        # Define Measurement Parameters
        self.R = MeasurementModel.R
        self.Q = MeasurementModel.Q
        
        
        # Prediction Step
        x_pred, P_pred = predict(self, est_state, P, u, dt)

        if isinstance(meas_state, str):
            x_hat = x_pred
            P = P_pred
        else:
            self.H = MeasurementModel.BuildMeasureJacob(est_state)
            # Update Step
            x_hat, P = update(self, x_pred, P_pred, meas_state)
            


        return x_hat, P

        


        
        
        