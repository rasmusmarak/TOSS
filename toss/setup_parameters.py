# Core packages
from math import pi
import numpy as np

# Load required modules
from utilities.load_default_cfg import load_default_cfg
from mesh.mesh_utility import create_mesh
from trajectory.equations_of_motion import setup_spin_axis


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

    # Load default constants value
    args = load_default_cfg()

    # Setup additional body properties
    args.body.spin_velocity = (2*pi)/args.body.spin_period
    args.body.spin_axis = setup_spin_axis(args)

    # Setup additional problem properties
    args.problem.squared_volume_inner_bounding_sphere = (4/3) * pi * (args.problem.radius_inner_bounding_sphere**3)
    args.problem.squared_volume_outer_bounding_sphere = (4/3) * pi * (args.problem.radius_outer_bounding_sphere**3)
    args.problem.total_measurable_volume = args.problem.squared_volume_outer_bounding_sphere - args.problem.squared_volume_inner_bounding_sphere
    args.problem.maximal_measurement_sphere_volume = (4/3) * pi * (args.problem.maximal_measurement_sphere_radius**3)

    # Create mesh of body:
    args.mesh.body, args.mesh.vertices, args.mesh.faces, args.mesh.largest_body_protuberant = create_mesh()

    # Defining the state variable and its boundaries (parameter space):
    #   state: [a, e, o, w, i, ea, tm, dvx, dvy, dvz]
    #
    # Declerations:
    #   a   : Semi-major axis
    #   e   : Eccentricity (e=[0,1]).
    #   o   : Right ascension of ascending node (o=[0,2*pi])
    #   w   : Argument of periapsis (w=[0,2*pi])
    #   i   : Inclination (i=[0,pi])
    #   ea  : Eccentric anomaly (ea=[0,2*pi])
    #   tm  : Time of impulsive maneuver ([seconds])
    #   dvx : Impulsive maneuver in x-axis
    #   dvy : Impulsive maneuver in y-axis
    #   dvz : Impulsive maneuver in z-axis
    #
    # NOTE: (Initial position and velocity are defined with osculating 
    #        orbital elements, i.e the first six parameters of the state vector)

    # Orbital elements
    a = [5000, 15000] 
    e = [0, 1]        
    o = [0, 2*pi]
    w = [0, 2*pi]     
    i = [0, pi]       
    ea = [0, 2*pi]

    # Impulsive Maneuver    
    tm = [(args.problem.start_time + 100), (args.problem.final_time - 100)]
    dvx = [-1, 1]
    dvy = [-1, 1]
    dvz = [-1, 1]
    
    # Generate boundary state vectors:
    lower_bounds = np.concatenate(([a[0], e[0], i[0], o[0], w[0], ea[0]], [tm[0], dvx[0], dvy[0], dvz[0]]*args.problem.number_of_maneuvers), axis=None)
    upper_bounds = np.concatenate(([a[1], e[1], i[1], o[1], w[1], ea[1]], [tm[1], dvx[1], dvy[1], dvz[1]]*args.problem.number_of_maneuvers), axis=None)

    return args, lower_bounds, upper_bounds