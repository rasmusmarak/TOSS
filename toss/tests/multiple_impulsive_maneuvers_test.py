""" This test checks whether or not the integration is performed correctly """
import sys
sys.path.append("../..")

# Import required modules
from toss.trajectory.equations_of_motion import compute_motion, setup_spin_axis
from toss.mesh.mesh_utility import create_mesh
from toss.trajectory.compute_trajectory import compute_trajectory
from toss.trajectory.trajectory_tools import get_trajectory_adaptive_step

# Core packages
from dotmap import DotMap
from math import pi
import numpy as np
import pykep as pk


def test_multiple_impulsive_maneuvers():

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

    # Arguments concerning bounding spheres
    args.problem.radius_inner_bounding_sphere = 4000      # Radius of spherical risk-zone for collision with celestial body [m]
    args.problem.measurement_period = 2500 # Period for when a measurement sphere is recognized and managed. Unit: [seconds]
    
    # Create mesh of body.
    args.mesh.body, args.mesh.vertices, args.mesh.faces, args.mesh.largest_body_protuberant = create_mesh()

    # Osculating orbital elements from initial state (position and velocity)
    x_cartesian = [-1.36986549e+03, -4.53113817e+03, -8.41816487e+03, -1.23505256e-01, -1.59791505e-01, 2.21471017e-01]
    x_osculating_elements = np.array(pk.ic2par(r=x_cartesian[0:3], v=x_cartesian[3:6], mu=args.body.mu))

    # Define magnitude of impulsive maneuver (all maneuvers are equivalent)
    dv_x = 0.1
    dv_y = 0.1
    dv_z = 0.1

    # Reference values from previous working state of code.
    #   (3x9) Array containing the final position (columnwise) of
    #   trajectories utilizing 0 to 8 number of maneuvers.
    final_positions_array = np.array([[3072.16679162, 23899.25828294, 12722.26422248, 2818.09785781, -3638.40268304, 797.26508049, -14928.16441209, -15143.95182972, -14184.09393631],
                            [-245.74091817, -3379.64207396, -4843.6805795, -4581.6197305, -3053.12842506, -4482.49904423, 1901.96283654, 8309.41455834, 10878.30074835],
                            [-9032.88999817, -17880.05036874, -19114.45554266, -16635.79018758, -10476.63314649, -10586.81716361, 14173.8414283, 15686.29468342, 27489.51165473]])

    # Generate a trajectory for each set of maneuvers and compare with reference array.
    max_number_of_maneuvers = 8
    for number_of_maneuvers in range(0,max_number_of_maneuvers+1):
        chromosome = x_osculating_elements
        args.problem.number_of_maneuvers = number_of_maneuvers

        if number_of_maneuvers > 0:
            # Define the chromosome corresponding to a specified number of maneuvers
            time_of_maneuver = 0
            for maneuver in range(1, number_of_maneuvers+1):
                time_of_maneuver += 5000
                chromosome = np.concatenate((chromosome, [time_of_maneuver, dv_x, dv_y, dv_z]), axis=None)
            
        # Compute trajectory via numerical integration as in UDP.
        _, list_of_trajectory_objects, _ = compute_trajectory(chromosome, args, compute_motion)

        # Get integration info:
        positions, _ = get_trajectory_adaptive_step(list_of_trajectory_objects)

        # Check if compute_trajectory still produces the same trajectories.
        assert all(np.isclose(final_positions_array[:, number_of_maneuvers],positions[0:3, -1],rtol=1e-5, atol=1e-5))