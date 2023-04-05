""" This test checks whether or not the integration is performed correctly """
import sys
sys.path.append("..")
sys.path.append("../..")

import os
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

# Import required modules
from trajectory.equations_of_motion import compute_motion, setup_spin_axis
import mesh.mesh_utility as mesh_utility
from trajectory.compute_trajectory import compute_trajectory
from trajectory.trajectory_tools import get_trajectory_fixed_step
from fitness.fitness_functions import target_altitude_distance, close_distance_penalty, far_distance_penalty, covered_volume, covered_volume_far_distance_penalty 

# Core packages
from dotmap import DotMap
from math import pi
import numpy as np
import pykep as pk


def get_parameters():
    """Returns parameters used by compute trajectory and fitness functions.

    Returns:
        args (dotmap.Dotmap): Dotmap with parameters used for the tests.
    """
    args = DotMap(
        body = DotMap(_dynamic=False),
        integrator = DotMap(_dynamic=False),
        problem = DotMap(_dynamic=False),
        mesh = DotMap(_dynamic=False),
        _dynamic=False)

    # Setup body parameters
    args.body.density = 533                  # https://sci.esa.int/web/rosetta/-/14615-comet-67p
    args.body.mu = 665.666                   # Gravitational parameter for 67P/C-G
    args.body.declination = 64               # [degrees] https://sci.esa.int/web/rosetta/-/14615-comet-67p
    args.body.right_ascension = 69           # [degrees] https://sci.esa.int/web/rosetta/-/14615-comet-67p
    args.body.spin_period = 12.06*3600       # [seconds] https://sci.esa.int/web/rosetta/-/14615-comet-67p
    args.body.spin_velocity = (2*pi)/args.body.spin_period
    args.body.spin_axis = setup_spin_axis(args)

    # Setup specific integrator parameters:
    args.integrator.algorithm = 3
    args.integrator.dense_output = True
    args.integrator.rtol = 1e-12
    args.integrator.atol = 1e-12

    # Setup problem parameters
    args.problem.start_time = 0                     # Starting time [s]
    args.problem.final_time = 20*3600.0             # Final time [s]
    args.problem.initial_time_step = 600            # Initial time step size for integration [s]
    args.problem.activate_event = True              # Event configuration (0 = no event, 1 = collision with body detection)
    args.problem.number_of_maneuvers = 0 
    args.problem.measurement_period = 2500
    args.problem.target_squared_altitude = 8000**2  # Target altitude squared [m]

    # Arguments concerning bounding spheres
    args.problem.measurement_period = 2500                # Period for when a measurement sphere is recognized and managed. Unit: [seconds]
    args.problem.radius_inner_bounding_sphere = 4000      # Radius of spherical risk-zone for collision with celestial body [m]
    args.problem.radius_outer_bounding_sphere = 10000
    args.problem.squared_volume_inner_bounding_sphere = (4/3) * pi * (args.problem.radius_inner_bounding_sphere**3)
    args.problem.squared_volume_outer_bounding_sphere = (4/3) * pi * (args.problem.radius_outer_bounding_sphere**3)
    args.problem.measurable_volume = args.problem.squared_volume_outer_bounding_sphere - args.problem.squared_volume_inner_bounding_sphere

    # Create mesh of body.
    args.mesh.body, args.mesh.vertices, args.mesh.faces, args.mesh.largest_body_protuberant = mesh_utility.create_mesh()

    return args


def get_trajectory(args):
    """ Computes trajectory defined by a fixed time step for a given initial position.

    Args:
        args (dotmap.Dotmap): Dotmap with parameters used for computing trajectory.

    Returns:
        positions (np.ndarray): (3,N) Array of positions along the trajectory.
        timesteps (None): (N) Array of time values for each position.  
    """

    # Initial position for integration (in cartesian coordinates):
    x = [-1.36986549e+03, -4.53113817e+03, -8.41816487e+03, -1.23505256e-01, -1.59791505e-01, 2.21471017e-01, 0, 0, 0, 0]
    x_osculating_elements = pk.ic2par(r=x[0:3], v=x[3:6], mu=args.body.mu) #translate to osculating orbital element

    # Compute trajectory via numerical integration as in UDP.
    _, list_of_ode_objects, _ = compute_trajectory(x_osculating_elements, args, compute_motion)

    # Get states along computed trajectory:
    positions, timesteps = get_trajectory_fixed_step(args, list_of_ode_objects)

    return positions, timesteps


def test_covered_volume():
    """
    Test to verify that the fitness function for covered volume works as expected. 
    """
    # Get parameters
    args = get_parameters()

    # Get trajectory
    positions, timesteps = get_trajectory(args)

    # Compute volume ratio
    volume_ratio = covered_volume(args, positions, timesteps)

    # Position and timesteps from previous working results (in cartesian coordinates):
    previous_results = -0.0012630002281608665
    
    assert np.isclose(volume_ratio,previous_results,rtol=1e-5, atol=1e-5)


def test_target_altitude_distance():
    """
    Test to verify that the fitness function for target altitude works as expected. 
    """
    # Get parameters
    args = get_parameters()

    # Get trajectory
    positions, timesteps = get_trajectory(args)

    # Compute target altitude distance fitness. 
    fitness = target_altitude_distance(args, positions, timesteps)

    # Previous results:
    previous_fitness = 2.4414062622701264e-16

    assert np.isclose(fitness,previous_fitness,rtol=1e-5, atol=1e-5)


def test_close_distance_penalty():
    """
    Test to verify that the fitness function for close distance penalty works as expected. 
    """
    # Get parameters
    args = get_parameters()

    # Define points to evaluate
    list_of_positions = np.arange((args.problem.radius_inner_bounding_sphere-1000),(args.problem.radius_inner_bounding_sphere+1000), 100)

    # Compute close distance penalty
    penalty = close_distance_penalty(args, list_of_positions, None)

    # Previous penalty
    previous_penalty = 0.6086956521739131

    assert np.isclose(penalty,previous_penalty,rtol=1e-5, atol=1e-5)


def test_far_distance_penalty():
    """
    Test to verify that the fitness function for far distance penalty works as expected. 
    """
    # Get parameters
    args = get_parameters()

    # Define points to evaluate
    list_of_positions = np.arange((args.problem.radius_outer_bounding_sphere-1000),(args.problem.radius_outer_bounding_sphere+1000), 100)

    # Compute close distance penalty
    penalty = far_distance_penalty(args, list_of_positions, None)
    
    # Previous penalty
    previous_penalty = 0.31664001346687987
    assert np.isclose(penalty,previous_penalty,rtol=1e-5, atol=1e-5)


def test_covered_volume_far_distance_penalty():
    """
    Test to verify that the aggregate fitness of covered volume and far distance penalty work as expected. 
    """
    # Get parameters
    args = get_parameters()

    # Define points to evaluate
    list_of_positions = np.arange((args.problem.radius_outer_bounding_sphere-1000),(args.problem.radius_outer_bounding_sphere+1000), 100)

    # Compute close distance penalty
    penalty = far_distance_penalty(args, list_of_positions, None)

    # Get trajectory
    positions, timesteps = get_trajectory(args)

    # Compute volume ratio
    volume_ratio = covered_volume(args, positions, timesteps)

    aggregate_fitness = volume_ratio + penalty
    previous_aggregate_fitness = 0.315377013238719

    assert np.isclose(aggregate_fitness,previous_aggregate_fitness,rtol=1e-5, atol=1e-5)