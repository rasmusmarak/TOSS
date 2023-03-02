# General
import numpy as np
from typing import Union

# For computing trajectory
from Trajectory import Trajectory

# For orbit representation (reference frame)
import pykep as pk

# Class representing UDP
class udp_initial_condition:
    """ 
    Sets up the user defined problem (udp) for use with pygmo.
    The object holds attributes in terms of variables and constants that
    are used for trajectory propagation. 
    The methods of the class defines the objective function for the optimization problem,
    boundaries for the state variables and computation of the fitness value for a given intial state. 
    """

    def __init__(self, body_args, target_altitude, final_time, start_time, time_step, lower_bounds, upper_bounds, algorithm, radius_bounding_sphere):
        """ Setup udp attributes.

        Args:
            body_args (dotmap.DotMap): Paramteers relating to the celestial body:
                density (float): Body density of celestial body.
                mu (float): Gravitational parameter for celestial body.
                declination (float): Declination angle of spin axis.
                right_ascension (float): Right ascension angle of spin axis.
                spin_period (float): Rotational period around spin axis of the body.
            target_altitude (float): Target altitude for satellite trajectory. 
            final_time (float): Final time for integration.
            start_time (float): Start time for integration of trajectory (often zero)
            time_step (float): Step size for integration. 
            lower_bounds (np.ndarray): Lower bounds for domain of initial state.
            upper_bounds (np.ndarray): Upper bounds for domain of initial state. 
            algorithm (int): User defined algorithm of choice
            radius_bounding_sphere (float)_: Radius for the bounding sphere around mesh.
        """

        # Setup equations of motion class
        self.trajectory = Trajectory(body_args, final_time, start_time, time_step, algorithm, radius_bounding_sphere)

        # Assertions:
        assert target_altitude > 0
        assert all(np.greater(upper_bounds, lower_bounds))

        # Assertions:
        assert body_args.mu > 0
        assert target_altitude > 0
        assert all(np.greater(upper_bounds, lower_bounds))

        # Additional hyperparameters
        self.body_args = body_args
        self.target_altitude = target_altitude     
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds


    def fitness(self, x: np.ndarray) -> float:
        """ fitness evaluates the proximity of the satallite to target altitude.

        Args:
            x (np.ndarray): State vector containing values for position and velocity of satelite in three dimensions. 

        Returns:
            fitness value (_float_): Difference between squared values of current and target altitude of satellite.
        """
        print("VALUEEEE: ", self.body_args.mu)
        # Convert osculating orbital elements to cartesian for integration
        r, v = pk.par2ic(E=x, mu=self.body_args.mu)
        r = np.array(r)
        v = np.array(v)
        x_cartesian = np.concatenate((r,v), axis=None)

        # Integrate trajectory
        _, squared_altitudes, collision_detected = self.trajectory.integrate(x_cartesian) #np.array(x_cartesian)

        # Define fitness penalty in the event of at least one collision along the trajectory
        if collision_detected == True:
            collision_penalty = 1e30
        else:
            collision_penalty = 0

        # Compute fitness value for the integrated trajectory
        fitness_value = np.mean(np.abs(squared_altitudes-self.target_altitude)) + collision_penalty

        return [fitness_value]


    def get_bounds(self) -> Union[np.ndarray, np.ndarray]:
        """get_bounds returns upper and lower bounds for the domain of the state vector.

        Returns:
            lower_bounds (np.ndarray): Lower boundary values for the initial state vector.
            upper_bounds (np.ndarray): Lower boundary values for the initial state vector.
        """
        return (self.lower_bounds, self.upper_bounds)
