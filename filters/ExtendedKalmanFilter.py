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
         
        self.Q = np.identity(6)

        self.H = np.identity(6)
        self.R = np.identity(6)
        

    def main(self, x_est0, x_meas, P, u, dt):

        def predict(self, state, P, u, dt):
            # Propagate State
            F = linalg.expm((ClohessyWiltshire.A+np.matmul(ClohessyWiltshire.B, u))*dt)
            x_pred = np.matmul(F,state) + np.matmul(self.B, u)

            # Predict Covariance
            #print(F)
            P_pred = np.matmul(np.matmul(F,P),F.transpose()) + self.Q
            return x_pred, P_pred

        def update(self, state, P_prev, meas_state):
            # Preliminary Info
            v = meas_state-state
            S = np.matmul(np.matmul(self.H,P_prev),self.H.transpose())+np.matmul(np.matmul(self.H,self.R),self.H.transpose())
            K = np.matmul(np.matmul(P_prev, self.H.transpose()),linalg.inv(S))

            # Update State Estimation
            x_hat = state+np.matmul(K,v)

            # Update Covariance Matrix
            # P = (I-KH)P(I-KH)'+KRK'
            I_KH = np.identity(6) - np.matmul(K,self.H)
            P = np.matmul(np.matmul(I_KH, P_prev),I_KH.transpose())+np.matmul(np.matmul(K,self.R),K.transpose())

            return x_hat, P
        
        # Prediction Step
        x_pred, P_pred = predict(self, x_est0, P, u, dt)

        #    Update Step
        x_hat, P_hat = update(self, x_pred, P_pred, x_meas)

        return x_hat, P_hat

        


        
        
        