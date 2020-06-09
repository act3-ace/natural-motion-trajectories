import numpy as np 
import math 
import scipy.linalg


def get_lqr_gain(A,B,Q,R):
    """
	Solve the continuous time lqr controller.
   		dx/dt = A x + B u
   		cost = integral ( x.T*Q*x + u.T*R*u ) 
    """
    
	# Solve ARE 
    Xare = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

	# Compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R)*(B.T*Xare))
    
    # Change to Array 
    K = np.squeeze(np.asarray(K))
    
    return K



def wrap_to_pi(theta) :
    """
	Finds corresponding angle between [-pi,pi]. 
	*NOTE: Only works for [-3pi,3pi], for general case, use atan2(sin(theta), cos(theta)) argument 
    """
    
    if theta > math.pi :
        theta = theta - math.pi
    elif  theta < -math.pi :
        theta = theta + 2*math.pi 
    return theta 
