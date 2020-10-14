# Translational Motion Simulator for Spacecraft Docking Problem 

Created by: Mark Mote and Chris Hays

Mentor: Kerianne Hobbs

Approved for public release: distribution unlimited. Case Number 88ABW-2020-2696.

This package provides simulation, estimation, and control for the attitude-independent version of the spacecraft docking problem, as outlined in the *challenge problem* document.    

## Key Variables 
- The state of the system (__x__) is a 6x1 numpy array. The first three components represent x,y, and z position coordinates in Hill's frame, and the latter three the derivatives of these coordinates. 
- The control input (__u__) is a 3x1 numpy array representing the applied forces along x, y, and z axes. 


## Getting Started 

System parameters are specified in _parameters.py_. The simulation is run via the _main.py_ script. The _controller_, _filters_, etc. are implemented as classes and specified by indicating the desired files at the top of the _main_ script. Simulator options and parameters can be specified in the Set Up section. 


## Directories 

### _controllers_ 
Controllers map the state and time to a desired input force. Each controller is implemented as a class, with a special _main_ method that is called in the loop; do not modify the inputs or outputs of this method. Each controller class is named "Controller" and given its own script.


## Dependencies 
- Many of the _controller_ and _asif_ modules require optimization to be performed with Gurobi. Installation of Gurobi with Anaconda can be done according to: https://www.gurobi.com/gurobi-and-anaconda-for-linux/. You may request a free licence at https://www.gurobi.com/academia/academic-program-and-licenses/
