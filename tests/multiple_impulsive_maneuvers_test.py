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
    final_positions_array = np.array([[23396.35072792, 21102.29874799, 10404.01196436, 7734.0043386, 6752.21928096, 8582.69914392, 8035.67743936, 6432.07600505, 6804.9235045],
                            [-18424.93333458, -8392.30034922, -6587.23157002, 2715.22505198, 17778.20939727, 15524.79648208, 7618.86627987, 4778.86809123, 4137.88904256],
                            [-6497.11495845, -14579.83433402, -10327.88436248, -5723.57621234, -5010.90293584, -9043.10220864, -5713.86988532, 35.92893325, 4246.9741536]])

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