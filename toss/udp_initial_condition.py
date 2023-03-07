# General
import numpy as np
from typing import Union

# For computing trajectory
import trajectory_tools

# For computing the next state
import equations_of_motion

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

    def __init__(self, args, lower_bounds, upper_bounds):
        """ Setup udp attributes.

        Args:
            body_args (dotmap.DotMap): Paramteers relating to the celestial body:
                density (float): Body density of celestial body.
                mu (float): Gravitational parameter for celestial body.
                declination (float): Declination angle of spin axis.
                right_ascension (float): Right ascension angle of spin axis.
                spin_period (float): Rotational period around spin axis of the body.
            target_squared_altitude (float): Target altitude for satellite trajectory. 
            final_time (float): Final time for integration.
            start_time (float): Start time for integration of trajectory (often zero)
            time_step (float): Step size for integration. 
            lower_bounds (np.ndarray): Lower bounds for domain of initial state.
            upper_bounds (np.ndarray): Upper bounds for domain of initial state. 
            algorithm (int): User defined algorithm of choice
            radius_bounding_sphere (float)_: Radius for the bounding sphere around mesh.
        """

        # Declerations
        self.target_sq_alt = args.problem.target_squared_altitude
        mu = args.body.mu
        ti = args.problem.start_time
        tf = args.problem.final_time
        dt = args.problem.initial_time_step
        r_sphere = args.problem.radius_bounding_sphere
        largest_protuberant = args.mesh.largest_body_protuberant
        density = args.body.density
        dec = args.body.declination
        ra = args.body.right_ascension
        period = args.body.spin_period

        # Assertions:
        assert self.target_sq_alt > 0
        assert all(np.greater(upper_bounds, lower_bounds))
        assert mu > 0
        assert tf > ti
        assert dt <= tf - ti
        assert r_sphere > largest_protuberant
        assert density > 0
        assert dec >= 0
        assert ra >= 0
        assert period >= 0


        # Additional hyperparameters
        self.args = args
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds


    def fitness(self, x: np.ndarray) -> float:
        """ fitness evaluates the proximity of the satallite to target altitude.

        Args:
            x (np.ndarray): State vector containing values for position and velocity of satelite in #D cartesian coordinates. 

        Returns:
            fitness value (_float_): Difference between squared values of current and target altitude of satellite.
        """
        # Convert osculating orbital elements to cartesian for integration
        r, v = pk.par2ic(E=x, mu=self.args.body.mu)
        x_cartesian = np.array(r+v)

        # Integrate trajectory
        _, squared_altitudes, collision_detected = trajectory_tools.compute_trajectory(x_cartesian, self.args, equations_of_motion.compute_motion)

        # Define fitness penalty in the event of at least one collision along the trajectory
        if collision_detected == True:
            collision_penalty = 1e30
        else:
            collision_penalty = 0

        # Compute fitness value for the integrated trajectory
        fitness_value = np.mean(np.abs(squared_altitudes-self.target_sq_alt)) + collision_penalty

        return [fitness_value]


    def get_bounds(self) -> Union[np.ndarray, np.ndarray]:
        """get_bounds returns upper and lower bounds for the domain of the state vector.

        Returns:
            lower_bounds (np.ndarray): Lower boundary values for the initial state vector.
            upper_bounds (np.ndarray): Lower boundary values for the initial state vector.
        """
        return (self.lower_bounds, self.upper_bounds)