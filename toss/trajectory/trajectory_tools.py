# Core packages
import numpy as np
from typing import Union


def get_trajectory_adaptive_step(list_of_ode_objects: list) -> Union[np.ndarray, np.ndarray]:
    """ Returns the computed trajectory (state and time) as provided by DEsolver with adaptive step size.

    Args:
        list_of_ode_objects (list): List of OdeSystem integration objects (provided by DEsolver)

    Returns:
        states (np.ndarray): (6,N) Array containing spacecraft positions and velocities expressed in cartesian frame.
        timesteps (np.ndarray): (N) Array containing adaptive time steps correspondning to positions. 
    """
    object_idx = 0
    for object_idx, ode_object in enumerate(list_of_ode_objects):
        if object_idx == 0:
            states = np.transpose(ode_object.y)
            timesteps = ode_object.t
        else:
            states = np.hstack((states, np.transpose(ode_object.y)))
            timesteps = np.hstack((timesteps, ode_object.t))
            
    return states, timesteps


def get_trajectory_fixed_step(args, list_of_ode_objects: list) -> Union[np.ndarray, np.ndarray]:
    """Returns the computed trajectory (position and time) as provided by DEsolver but for a user-define fixed time-step.

    Args:
        args (dotmap.DotMap):
            problem:
                start_time (int): Start time of integration.
                final_time (int): Final time of integration.
                measurement_period (int): Period for which a measurment sphere is recognized and managed.
        list_of_ode_objects (list): List holding the OdeSystem trajectory object for each discretized integration interval.

    Returns:
        positions (np.ndarray): (3,N) Array containing satelite position epressed in cartesian frame.
        velocities (np.ndarray): (3,N) Array containing satelite velocities epressed in cartesian frame.
        timesteps (np.ndarray): (N) Array containing fixed time steps correspondning to positions. 
    """
    
    # Define times-axis with a fixed time step
    timesteps = np.arange(args.problem.start_time, args.problem.final_time, args.problem.measurement_period)

    # Get satellite positions at times defined in timesteps
    positions = np.empty((3,len(timesteps)), dtype=np.float64)
    velocities = np.empty((3,len(timesteps)), dtype=np.float64)
    start_idx = 0
    for ode_object in list_of_ode_objects:

        # Find nearest idx in time_step (end_time_idx) corresponding to ode_end_time defined in ode_object
        ode_end_time = ode_object.t[-1]
        end_time_idx = (np.abs(timesteps - ode_end_time)).argmin()
        if ode_end_time < timesteps[end_time_idx]:
            end_time_idx -= 1

        if start_idx == end_time_idx + 1:
            end_idx = end_time_idx + 2
        else: 
            end_idx = end_time_idx + 1

        #if ode_end_time != 0 and len(list_of_ode_objects) > 1:
        # Get positions and velocities using dense_output of ode_object
        ode_dense_output = np.transpose(ode_object._OdeSystem__sol(timesteps[start_idx:end_idx]))
        positions[:, start_idx:end_time_idx+1] = ode_dense_output[0:3,:]
        velocities[:, start_idx:end_time_idx+1] = ode_dense_output[3:6,:]
        start_idx = end_time_idx + 1

        # Stop criteria if desired points have been computed before iterating through all ode objects
        if len(timesteps)-1 < start_idx:
            break

    return positions, velocities, timesteps