# Core packages
import numpy as np
from math import pi

def setup_initial_state_domain(initial_condition, start_time, final_time, number_of_maneuvers, number_of_spacecrafts, state):
    """
    This is a function that returns the upper and lower bounds corresponding 
    to the initial state space. The configurations of the bounds depend on the
    desired structure of the chromosome. For instance, if the problem assumes a given
    initial state (position and velocity), the bounds will only consider maneuvers.
    Otherwise, the initial state will be concatenated with the maneuvers to define the bounds.

    NOTE: In this function, we use the following notations for maneuvers:
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
        state (dotMap): Parameters constricting the initial state space.

    Returns: 
        lower_bounds (np.ndarray): Lower boundary values for the initial state vector.
        upper_bounds (np.ndarray): Lower boundary values for the initial state vector.    
    """
    # Setup time of maneuver variable (depends on mission duration)
    tm = [(start_time + 1), (final_time - 1)]

    # Optimizing initial position, initial velocity and maneuvers
    if len(initial_condition) == 0:
        lower_bounds = np.concatenate(([state.x_min, state.y_min, state.z_min, state.v_min, state.vx_min, state.vy_min, state.vz_min], [tm[0], state.dv_min, state.dvx_min, state.dvy_min, state.dvz_min]*number_of_maneuvers)*number_of_spacecrafts, axis=None)
        upper_bounds = np.concatenate(([state.x_max, state.y_max, state.z_max, state.v_max, state.vx_max, state.vy_max, state.vz_max], [tm[1], state.dv_max, state.dvx_max, state.dvy_max, state.dvz_max]*number_of_maneuvers)*number_of_spacecrafts, axis=None)

    # Optimizing initial velocity and maneuvers
    elif len(initial_condition) == (number_of_spacecrafts*3):
        lower_bounds = np.concatenate(([state.v_min, state.vx_min, state.vy_min, state.vz_min], [tm[0], state.dv_min, state.dvx_min, state.dvy_min, state.dvz_min]*number_of_maneuvers)*number_of_spacecrafts, axis=None)
        upper_bounds = np.concatenate(([state.v_max, state.vx_max, state.vy_max, state.vz_max], [tm[1], state.dv_max, state.dvx_max, state.dvy_max, state.dvz_max]*number_of_maneuvers)*number_of_spacecrafts, axis=None)

    # Optimizing only maneuvers
    else:
        lower_bounds = [tm[0], state.dv_min, state.dvx_min, state.dvy_min, state.dvz_min]*number_of_maneuvers*number_of_spacecrafts
        upper_bounds = [tm[1], state.dv_max, state.dvx_max, state.dvy_max, state.dvz_max]*number_of_maneuvers*number_of_spacecrafts

    return lower_bounds, upper_bounds