# Core packages
import pygmo as pg
import numpy as np
from math import pi
from dotmap import DotMap

# For creating mesh
import mesh_utility

# For computing the next state
import equations_of_motion

# For optimization using pygmo
from udp_initial_condition import udp_initial_condition





def setup_parameters():
    """Set up of required hyperparameters for the optimization scheme. 

    Returns:

        body_args (dotmap.DotMap): Parameters related to the celestial body:
            density (float): Body density of celestial body.
            mu (float): Gravitational parameter for celestial body.
            declination (float): Declination angle of spin axis.
            right_ascension (float): Right ascension angle of spin axis.
            spin_period (float): Rotational period around spin axis of the body.
            spin_velocity (float): Angular velocity of the body's rotation.
            spin_axis (np.ndarray): The axis around which the body rotates.

        integrator_args (dotmap.DotMap): Specific parameters related to the integrator:
            algorithm (int): Integer representing specific integrator algorithm.
            dense_output (bool): Dense output status of integrator.
            rtol (float): Relative error tolerance for integration.
            atol (float): Absolute error tolerance for integration.

        problem_args (dotmap.DotMap): Parameters related to the problem:
            start_time (float): Start time (in seconds) for the integration of trajectory.
            final_time (float): Final time (in seconds) for the integration of trajectory.
            initial_time_step (float): Size of initial time step (in seconds) for integration of trajectory.
            target_squared_altitude (float): Squared value of the satellite's orbital target altitude.
            radius_bounding_sphere (float): Radius of the bounding sphere representing risk zone for collisions with celestial body.
            event (int): Event configuration (0 = no event, 1 = collision with body detection)
        
        lower_bounds (np.ndarray): Lower boundary values for the initial state vector.
        upper_bounds (np.ndarray): Lower boundary values for the initial state vector.
                
        population_size (int): Number of chromosomes to compare at each generation.
        number_of_generations (int): Number of generations for the genetic opimization.
    """
    args = DotMap()

    # Setup body parameters
    args.body.density = 533                  # https://sci.esa.int/web/rosetta/-/14615-comet-67p
    args.body.mu = 665.666                   # Gravitational parameter for 67P/C-G
    args.body.declination = 64               # [degrees] https://sci.esa.int/web/rosetta/-/14615-comet-67p
    args.body.right_ascension = 69           # [degrees] https://sci.esa.int/web/rosetta/-/14615-comet-67p
    args.body.spin_period = 12.06*3600       # [seconds] https://sci.esa.int/web/rosetta/-/14615-comet-67p
    args.body.spin_velocity = (2*pi)/args.body.spin_period
    args.body.spin_axis = equations_of_motion.setup_spin_axis(args)

    # Setup specific integrator parameters:
    args.integrator.algorithm = 3
    args.integrator.dense_output = True
    args.integrator.rtol = 1e-12
    args.integrator.atol = 1e-12

    # Setup problem parameters
    args.problem.start_time = 0                     # Starting time [s]
    args.problem.final_time = 20*3600.0             # Final time [s]
    args.problem.initial_time_step = 600            # Initial time step size for integration [s]
    args.problem.radius_bounding_sphere = 4000      # Radius of spherical risk-zone for collision with celestial body [m]
    args.problem.target_squared_altitude = 8000**2  # Target altitude squared [m]
    args.problem.event = 1                          # Event configuration (0 = no event, 1 = collision with body detection)
    

    # Defining the parameter space for the optimization
    #   (Parameters are defined in osculating orbital elements)
    a = [5000, 15000] # Semi-major axis
    e = [0, 1]        # Eccentricity [0, 1]
    o = [0, 2*pi]     # Right ascension of ascending node [0,2*pi]
    w = [0, 2*pi]     # Argument of periapsis [0, 2*pi]
    i = [0, pi]       # Inclination [0, pi] 
    ea = [0, 2*pi]    # Eccentric anomaly [0, 2*pi]

    lower_bounds = np.array([a[0], e[0], i[0], o[0], w[0], ea[0]])
    upper_bounds = np.array([a[1], e[1], i[1], o[1], w[1], ea[1]])


    # Optimization parameters
    population_size = 7
    number_of_generations = 4
    number_of_islands = 4 # one per thread of the cpu

    return args, lower_bounds, upper_bounds, population_size, number_of_generations, number_of_islands

def pop_init(population_size, seed):

    #print("Retrieving user defined parameters...")
    args, lower_bounds, upper_bounds, _, _, _ = setup_parameters()

    # Creating the mesh (TetGen)
    #print("Creating the mesh...")
    args.mesh.body, args.mesh.vertices, args.mesh.faces, args.mesh.largest_body_protuberant = mesh_utility.create_mesh()

    # Setup User-Defined Problem (UDP)
    #print("Setting up the UDP...")
    udp = udp_initial_condition(args, lower_bounds, upper_bounds)
    prob = pg.problem(udp)

    # Setup optimization algorithm
    #print("Setting up the optimization algorithm...")
    assert population_size >= 7

    # Return population
    return pg.population(prob, population_size, seed=seed)