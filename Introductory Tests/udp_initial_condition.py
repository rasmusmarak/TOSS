# General
import numpy as np
from typing import Union

# For computing trajectory
from Trajectory import Trajectory


# Class representing UDP
class udp_initial_condition:
    """ 
    Sets up the user defined problem (udp) for use with pygmo.
    The object holds attributes in terms of variables and constants that
    are used for trajectory propagation. 
    The methods of the class defines the objective function for the optimization problem,
    boundaries for the state variables and computation of the fitness value for a given intial state. 
    """

    def __init__(self, body_density, target_altitude, final_time, start_time, time_step, lower_bounds, upper_bounds, algorithm, radius_bounding_sphere):
        """ Setup udp attributes.

        Args:
            body_density (_float_): Mass density of body of interest
            target_altitude (_float_): Target altitude for satellite trajectory. 
            final_time (_float_): Final time for integration.
            start_time (_float_): Start time for integration of trajectory (often zero)
            time_step (_float_): Step size for integration. 
            lower_bounds (_np.ndarray_): Lower bounds for domain of initial state.
            upper_bounds (_np.ndarray_): Upper bounds for domain of initial state. 
            algorithm (_int_): User defined algorithm of choice
            radius_bounding_sphere (_float_)_: Radius for the bounding sphere around mesh.
        """

        # Setup equations of motion class
        self.trajectory = Trajectory(body_density, final_time, start_time, time_step, algorithm, radius_bounding_sphere)

        # Assertions:
        assert target_altitude > 0
        assert all(np.greater(upper_bounds, lower_bounds))

        # Additional hyperparameters
        self.target_altitude = target_altitude     
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def fitness(self, x: np.ndarray) -> float:
        """ fitness evaluates the proximity of the satallite to target altitude.

        Args:
            x (_np.ndarray_): State vector containing values for position and velocity of satelite in three dimensions. 

        Returns:
            fitness value (_float_): Difference between squared values of current and target altitude of satellite.
        """
        # Integrate trajectory
        _, squared_altitudes, collision_penalty = self.trajectory.integrate(np.array(x))

        # Compute fitness value for the integrated trajectory
        fitness_value = np.mean(np.abs(squared_altitudes-self.target_altitude)) + collision_penalty

        return [fitness_value]


    def get_bounds(self) -> Union[np.ndarray, np.ndarray]:
        """get_bounds returns upper and lower bounds for the domain of the state vector.

        Returns:
            lower_bounds (_np.ndarray_): Lower boundary values for the initial state vector.
            upper_bounds (_np.ndarray_): Lower boundary values for the initial state vector.
        """
        return (self.lower_bounds, self.upper_bounds)
