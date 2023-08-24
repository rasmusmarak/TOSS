""" This test checks whether or not the integration is performed correctly """
# Import required modules
from toss import compute_motion
from toss import create_mesh
from toss import compute_trajectory
from toss import get_trajectory_fixed_step
from toss import target_altitude_distance, close_distance_penalty, far_distance_penalty, covered_volume, total_covered_volume
from toss import setup_parameters

# Core packages
import numpy as np

def get_parameters():
    """Returns parameters used by compute trajectory and fitness functions.

    Returns:
        args (dotmap.Dotmap): Dotmap with parameters used for the tests.
    """
    # Load parameters from default cfg.
    args = setup_parameters()

    # Adjust for test-specific parameters:
    args.problem.start_time = 0                     # Starting time [s]
    args.problem.final_time = 20*3600.0             # Final time [s]
    args.problem.initial_time_step = 600            # Initial time step size for integration [s]
    args.problem.activate_event = True              # Event configuration (0 = no event, 1 = collision with body detection)
    args.problem.number_of_maneuvers = 0 
    args.problem.target_squared_altitude = 8000**2  # Target altitude squared [m]
    args.problem.activate_rotation = True
    args.problem.penalty_scaling_factor = 1         # Scales the magnitude of the fixed-valued maximal velocity, and therefore also the grid spacing.
    args.problem.measurement_period = 100                # Period for when a measurement sphere is recognized and managed. Unit: [seconds]
    args.problem.radius_inner_bounding_sphere = 4000      # Radius of spherical risk-zone for collision with celestial body [m]
    args.problem.radius_outer_bounding_sphere = 10000
    args.mesh.mesh_path = "3dmeshes/churyumov-gerasimenko_llp.pk"
    args.mesh.body, args.mesh.vertices, args.mesh.faces, args.mesh.largest_body_protuberant = create_mesh(args.mesh.mesh_path)

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
    #   NOTE: The initial state vector is structured as x = [rx, ry, rz, v_magnitude, vx, vy, vz]
    #         And represents the optimal initial position found for the single spacecraft case 
    #         presented in: https://doi.org/10.48550/arXiv.2306.01602. 
    x = np.array([-135.13402075, -4089.53592604, 6050.17636635, 2.346971623591584122e-01, 6.959989121956766667e-01, -9.249848356174805719e-01, 7.262727928440093628e-01])

    # Compute trajectory via numerical integration as in UDP.
    _, list_of_ode_objects, _ = compute_trajectory(x, args, compute_motion)

    # Get states along computed trajectory:
    positions, _, timesteps = get_trajectory_fixed_step(args, list_of_ode_objects)

    return positions, timesteps


def test_covered_volume():
    """
    Test to verify that the fitness function for covered volume by a single satellite works as expected. 
    """
    # Get parameters
    args = get_parameters()

    # Get trajectory
    positions, _ = get_trajectory(args)

    # Compute volume ratio
    volume_ratio = covered_volume(args.problem.maximal_measurement_sphere_volume, positions)

    # Position and timesteps from previous working results (in cartesian coordinates):
    previous_results = -0.24490757596526147
    assert np.isclose(volume_ratio,previous_results,rtol=1e-5, atol=1e-5)


def test_target_altitude_distance():
    """
    Test to verify that the fitness function for target altitude works as expected. 
    """
    # Get parameters
    args = get_parameters()

    # Get trajectory
    positions, _ = get_trajectory(args)

    # Compute target altitude distance fitness. 
    fitness = target_altitude_distance(args.problem.target_squared_altitude, positions)

    # Previous results:
    previous_fitness = 1.416195377466741
    assert np.isclose(fitness,previous_fitness,rtol=1e-5, atol=1e-5)


def test_close_distance_penalty():
    """
    Test to verify that the fitness function for close distance penalty works as expected. 
    """
    # Get parameters
    args = get_parameters()

    # Define points to evaluate
    list_of_positions = np.arange((args.problem.radius_inner_bounding_sphere-1000),(args.problem.radius_inner_bounding_sphere+1000), 100)
    positions = np.zeros((3, len(list_of_positions)))
    positions[0,:] = list_of_positions

    # Compute close distance penalty
    penalty = close_distance_penalty(args.problem.radius_inner_bounding_sphere, positions, args.problem.penalty_scaling_factor)

    # Previous penalty
    previous_penalty = 0.8132882808488928
    assert np.isclose(penalty,previous_penalty,rtol=1e-5, atol=1e-5)


def test_far_distance_penalty():
    """
    Test to verify that the fitness function for far distance penalty works as expected. 
    """
    # Get parameters
    args = get_parameters()

    # Define points to evaluate
    list_of_positions = np.arange((args.problem.radius_outer_bounding_sphere-1000),(args.problem.radius_outer_bounding_sphere+1000), 100)
    positions = np.zeros((3, len(list_of_positions)))
    positions[0,:] = list_of_positions

    # Compute far distance penalty
    penalty = far_distance_penalty(args.problem.radius_outer_bounding_sphere, positions, args.problem.penalty_scaling_factor)
    
    # Previous penalty
    previous_penalty = 0.5667412838561734
    assert np.isclose(penalty,previous_penalty,rtol=1e-5, atol=1e-5)


def test_total_covered_volume():
    """
    Test to verify that the fitness function for total covered volume works as expected. 
    """
    # Get parameters
    args = get_parameters()

    # Get trajectory
    positions, _ = get_trajectory(args)

    # Compute volume ratio
    volume_ratio = total_covered_volume(args.problem.total_measurable_volume, positions)

    # Previous ratio
    previous_ratio = -4.338256930189303e-06
    assert np.isclose(volume_ratio,previous_ratio,rtol=1e-5, atol=1e-5)

