# Core packages
import numpy as np
from typing import Union

# Import required modules
from toss.trajectory.compute_trajectory import compute_trajectory
from toss.trajectory.equations_of_motion import compute_motion 
from toss.trajectory.trajectory_tools import get_trajectory_fixed_step
from toss.fitness.fitness_function_enums import FitnessFunctions
from toss.fitness.fitness_functions import get_fitness

# Class representing UDP 
class udp_initial_condition:
    """ 
    Sets up the user defined problem (udp) to be optimized with pygmo.
    The object holds attributes that are mainly used for propagating the trajectory. 
    The methods of the class mainly defines the objective function (fitness) for the 
    optimization problem as well as the domain of the intial state vector. 
    """

    def __init__(self, args, lower_bounds, upper_bounds):
        """ Setup udp attributes.

        Args:
            args (dotmap.Dotmap)
                body: Parameters related to the celestial body:
                    density (float): Mass density of celestial body.
                    mu (float): Gravitational parameter for celestial body.
                    declination (float): Declination angle of spin axis.
                    right_ascension (float): Right ascension angle of spin axis.
                    spin_period (float): Rotational period around spin axis of the body.
                    spin_velocity (float): Angular velocity of the body's rotation.
                    spin_axis (np.ndarray): The axis around which the body rotates.
                integrator: Specific parameters related to the integrator:
                    algorithm (int): Integer representing specific integrator algorithm.
                    dense_output (bool): Dense output status of integrator.
                    rtol (float): Relative error tolerance for integration.
                    atol (float): Absolute error tolerance for integration.
                problem: Parameters related to the problem:
                    start_time (float): Start time (in seconds) for the integration of trajectory.
                    final_time (float): Final time (in seconds) for the integration of trajectory.
                    initial_time_step (float): Size of initial time step (in seconds) for integration of trajectory.
                    target_squared_altitude (float): Squared value of the satellite's orbital target altitude.
                    activate_event (bool): Event configuration (0 = no event, 1 = collision with body detection).
                    number_of_maneuvers (int): Number of possible maneuvers.
                    radius_inner_bounding_sphere (float): Radius of the inner bounding sphere representing risk zone for collisions with celestial body.
                    radius_outer_bounding_sphere (float): Radius of the outer bounding sphere.
                    squared_volume_inner_bounding_sphere (float): squared volume of the inner bounding sphere.
                    squared_volume_outer_bounding_sphere (float): squared volume of the outer bounding sphere.
                    measurable_squared_volume (float): Total measurable volume, i.e:    V_outer_sphere - V_inner_sphere.
                    measurement_period (int): Period for which a measurment sphere is recognized and managed.
                mesh:
                    vertices (np.ndarray): Array containing all points on mesh.
                    faces (np.ndarray): Array containing all triangles on the mesh.
            lower_bounds (np.ndarray): Lower bounds for domain of initial state.
            upper_bounds (np.ndarray): Upper bounds for domain of initial state. 
        """

        # Declerations
        self.target_sq_alt = args.problem.target_squared_altitude
        mu = args.body.mu
        ti = args.problem.start_time
        tf = args.problem.final_time
        dt = args.problem.initial_time_step
        r_inner_sphere = args.problem.radius_inner_bounding_sphere
        r_outer_sphere = args.problem.radius_outer_bounding_sphere
        v_inner_sphere =  args.problem.squared_volume_inner_bounding_sphere
        v_outer_sphere =  args.problem.squared_volume_outer_bounding_sphere
        v_measurable = args.problem.measurable_squared_volume
        largest_protuberant = args.mesh.largest_body_protuberant
        density = args.body.density
        dec = args.body.declination
        ra = args.body.right_ascension
        period = args.body.spin_period
        n_maneuvers = args.problem.number_of_maneuvers
        activate_events = args.problem.activate_event
        meaurement_period = args.problem.measurement_period
        fitness_list = args.problem.selected_fitness_functions

        # Assertions:
        assert self.target_sq_alt > 0
        assert all(np.greater(upper_bounds, lower_bounds))
        assert mu > 0
        assert tf > ti
        assert dt <= tf - ti
        assert r_inner_sphere > largest_protuberant
        assert density > 0
        assert dec >= 0
        assert ra >= 0
        assert period >= 0
        assert n_maneuvers >= 0
        assert isinstance(n_maneuvers, int)
        assert isinstance(activate_events, bool)
        assert (r_outer_sphere > r_inner_sphere)
        assert (v_outer_sphere > v_inner_sphere)
        assert (v_measurable > 0)
        assert (meaurement_period > 0)
        assert (len(fitness_list) > 0)


        # Additional hyperparameters
        self.args = args
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds


    def fitness(self, x: np.ndarray) -> float:
        """ Evaluates and returns fitness of the computed trajectory.

        Args:
            x (np.ndarray): Initial state vector. 

        Returns:
            fitness (_float_): Evaluated fitness for user-specified fitness-function.
        """

        # Compute trajectory
        collision_detected, list_of_ode_objects, _ = compute_trajectory(x, self.args, compute_motion)

        # If collision detected => unfeasible trajectory
        if collision_detected:
            fitness = 1e30
            return [fitness]
        
        # Get positions on trajectory for a fixed time-step
        positions, timesteps = get_trajectory_fixed_step(self.args, list_of_ode_objects)

        # Compute fitness:
        chosen_fitness_function = FitnessFunctions.CoveredVolumeFarDistancePenalty
        fitness = get_fitness(chosen_fitness_function, self.args, positions, timesteps)
        return [fitness]


    def get_bounds(self) -> Union[np.ndarray, np.ndarray]:
        """Returns upper and lower bounds for the domain of the initial state vector.

        Returns:
            lower_bounds (np.ndarray): Lower boundary values for the initial state vector.
            upper_bounds (np.ndarray): Lower boundary values for the initial state vector.
        """
        return (self.lower_bounds, self.upper_bounds)
    