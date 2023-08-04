""" This test checks whether or not the integration is performed correctly """

# Import required modules
from toss import compute_motion
from toss import create_mesh
from toss import compute_trajectory
from toss import get_trajectory_adaptive_step
from toss import setup_parameters

# Core packages
import numpy as np

def test_multiple_impulsive_maneuvers():

    # Load parameters from default cfg.
    args = setup_parameters()

    # Adjust for test-specific parameters:
    args.problem.start_time = 0                     # Starting time [s]
    args.problem.final_time = 20*3600.0             # Final time [s]
    args.problem.initial_time_step = 600            # Initial time step size for integration [s]
    args.problem.activate_event = True              # Event configuration (0 = no event, 1 = collision with body detection)
    args.problem.activate_rotation = True
    args.problem.radius_inner_bounding_sphere = 4000      # Radius of spherical risk-zone for collision with celestial body [m]
    args.problem.measurement_period = 2500 # Period for when a measurement sphere is recognized and managed. Unit: [seconds]
    args.mesh.mesh_path = "3dmeshes/churyumov-gerasimenko_llp.pk"
    args.mesh.body, args.mesh.vertices, args.mesh.faces, args.mesh.largest_body_protuberant = create_mesh(args.mesh.mesh_path)

    # Initial position for integration (in cartesian coordinates):
    #   NOTE: The initial state vector is structured as x = [rx, ry, rz, v_magnitude, vx, vy, vz]
    #         And represents the optimal initial position found for the single spacecraft case 
    #         presented in: https://doi.org/10.48550/arXiv.2306.01602. 
    x = np.array([-135.13402075, -4089.53592604, 6050.17636635, 2.346971623591584122e-01, 6.959989121956766667e-01, -9.249848356174805719e-01, 7.262727928440093628e-01])

    # Define magnitude of impulsive maneuver (all maneuvers are equivalent)
    dv_magnitude = 1
    dv_x = 0.1
    dv_y = 0.1
    dv_z = 0.1

    # Reference values from previous working state of code.
    #   (3x9) Array containing the final position (columnwise) of
    #   trajectories utilizing 0 to 8 number of maneuvers.
    final_positions_array = np.array([[-2297.36099208, -4729.29498965, 1138.98252462, 5777.75704377, 7760.2096581, 7475.5098274, 6279.87939984, 4461.39179979, 3828.35610737],
                            [-24013.70663895, -15879.92879791, -13681.18083499, -11203.97423552, -8021.88303958, -5788.31227738, -4031.44052364, -2252.39023405, -1453.03453712],
                            [6382.94419971, 353.89828333, 1472.61874213, 1424.21906177, 693.5329445, 95.22638442, 181.39058788, 2460.25118929, 6033.07983753]])

    # Generate a trajectory for each set of maneuvers and compare with reference array.
    max_number_of_maneuvers = 8
    for number_of_maneuvers in range(0,max_number_of_maneuvers+1):
        chromosome = x
        args.problem.number_of_maneuvers = number_of_maneuvers

        if number_of_maneuvers > 0:
            # Define the chromosome corresponding to a specified number of maneuvers
            time_of_maneuver = 0
            for maneuver in range(1, number_of_maneuvers+1):
                time_of_maneuver += 5000
                chromosome = np.concatenate((chromosome, [time_of_maneuver, dv_magnitude, dv_x, dv_y, dv_z]), axis=None)
            
        # Compute trajectory via numerical integration as in UDP.
        _, list_of_trajectory_objects, _ = compute_trajectory(chromosome, args, compute_motion)

        # Get integration info:
        positions, _ = get_trajectory_adaptive_step(list_of_trajectory_objects)

        # Check if compute_trajectory still produces the same trajectories.
        assert all(np.isclose(final_positions_array[:, number_of_maneuvers],positions[0:3, -1],rtol=1e-5, atol=1e-5))