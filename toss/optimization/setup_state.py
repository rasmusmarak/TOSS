# Core packages
import numpy as np
from math import pi

def setup_initial_state_domain(initial_condition, start_time, final_time, number_of_maneuvers, number_of_spacecrafts):
    """
    This is a function that returns the upper and lower bounds corresponding 
    to the initial state variable to be optimized. The configuration of the state variable
    depends on the purpose of the optimization. For instance, if we are given a predefined
    intial state, expressed in osculating orbital elements, the state variable will be 
    defined by the maneuvers of each spacecraft only. Otherwise, we concatenate the 
    initial state boundaries with the corresponding bounds for impulsive maneuvers. 

    NOTE: In this function, we use the following notations:
           a   : Semi-major axis
           e   : Eccentricity (e=[0,1]).
           o   : Right ascension of ascending node (o=[0,2*pi])
           w   : Argument of periapsis (w=[0,2*pi])
           i   : Inclination (i=[0,pi])
           ea  : Eccentric anomaly (ea=[0,2*pi])
           tm  : Time of impulsive maneuver ([seconds])
           dvx : Impulsive maneuver in x-axis
           dvy : Impulsive maneuver in y-axis
           dvz : Impulsive maneuver in z-axis
           
    args:
        initial_condition (np.ndarray): Initital state (position and velocity) expressed in osculating elements
        start_time (float): Start time (in seconds) for the integration of trajectory.
        final_time (float): Final time (in seconds) for the integration of trajectory.
        number_of_maneuvers (int): Number of maneuvers allowed per spacecraft.
        number_of_spacecrafts (int): Number of spacecrafts defined in our optimization problem.

    Returns: 
        lower_bounds (np.ndarray): Lower boundary values for the initial state vector.
        upper_bounds (np.ndarray): Lower boundary values for the initial state vector.    
    """

    # Define boundaries for osculating orbital elements
    a = [4000, 15000] #[5000, 15000] 
    e = [0, 1]        
    o = [0, 2*pi]
    w = [0, 2*pi]     
    i = [0, pi]       
    ea = [0, 2*pi]

    # Define boundaries for an impulsive Maneuver    
    tm = [(start_time + 1), (final_time - 1)]
    dvx = [-1, 1]
    dvy = [-1, 1]
    dvz = [-1, 1]

    # Optimizing initial state and maneuvers
    if len(initial_condition) == 0:
        lower_bounds = np.concatenate(([a[0], e[0], i[0], o[0], w[0], ea[0]], [tm[0], dvx[0], dvy[0], dvz[0]]*number_of_maneuvers)*number_of_spacecrafts, axis=None)
        upper_bounds = np.concatenate(([a[1], e[1], i[1], o[1], w[1], ea[1]], [tm[1], dvx[1], dvy[1], dvz[1]]*number_of_maneuvers)*number_of_spacecrafts, axis=None)

    # Optimizing only maneuvers
    else:
        lower_bounds = [tm[0], dvx[0], dvy[0], dvz[0]]*(number_of_maneuvers*number_of_spacecrafts)
        upper_bounds = [tm[1], dvx[1], dvy[1], dvz[1]]*(number_of_maneuvers*number_of_spacecrafts)


    return lower_bounds, upper_bounds