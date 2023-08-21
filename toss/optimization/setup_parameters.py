# Core packages
from math import pi
import numpy as np
import polyhedral_gravity as model

# Load required modules
from toss.utilities.load_default_cfg import load_default_cfg
from toss.mesh.mesh_utility import create_mesh
from toss.trajectory.equations_of_motion import setup_spin_axis
from toss.fitness.fitness_function_utils import create_spherical_tensor_grid

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
        
        optimization_args:
            population_size (int): Number of chromosomes to compare at each generation.
            number_of_generations (int): Number of generations for the genetic opimization.
    """

    # Load default constants value
    args = load_default_cfg()

    # Setup additional problem properties
    args.problem.squared_volume_inner_bounding_sphere = (4/3) * pi * (args.problem.radius_inner_bounding_sphere**3)
    args.problem.squared_volume_outer_bounding_sphere = (4/3) * pi * (args.problem.radius_outer_bounding_sphere**3)
    args.problem.total_measurable_volume = args.problem.squared_volume_outer_bounding_sphere - args.problem.squared_volume_inner_bounding_sphere
    args.problem.maximal_measurement_sphere_volume = (4/3) * pi * (args.problem.maximal_measurement_sphere_radius**3)

    # Setup additional body properties
    args.body.spin_velocity = (2*pi)/args.body.spin_period
    args.body.spin_axis = setup_spin_axis(args)

    # Create mesh of body and polyhedral object:
    args.mesh.body, args.mesh.vertices, args.mesh.faces, args.mesh.largest_body_protuberant = create_mesh(args.mesh.mesh_path)
    args.mesh.evaluable = model.GravityEvaluable((args.mesh.vertices, args.mesh.faces), args.body.density)

    # Setup initial boolean tensor representing the spherical grid approximation of the body's gravity field
    args.problem.fixed_velocity = np.array([args.problem.sample_vx, args.problem.sample_vy, args.problem.sample_vz])
    args.problem.tensor_grid_r, args.problem.tensor_grid_theta, args.problem.tensor_grid_phi, args.problem.bool_tensor, args.problem.tensor_weights = create_spherical_tensor_grid(args.problem.measurement_period, args.problem.radius_inner_bounding_sphere, args.problem.radius_outer_bounding_sphere, args.problem.max_velocity_scaling_factor, args.problem.fixed_velocity)
    
    return args