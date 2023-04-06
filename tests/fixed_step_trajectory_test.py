""" This test checks whether or not the integration is performed correctly """

# Import required modules
from toss.trajectory.equations_of_motion import compute_motion, setup_spin_axis
from toss.mesh.mesh_utility import create_mesh
from toss.trajectory.compute_trajectory import compute_trajectory
from toss.trajectory.trajectory_tools import get_trajectory_fixed_step

# Core packages
from dotmap import DotMap
from math import pi
import numpy as np
import pykep as pk


def test_integration():
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

    # Arguments concerning bounding spheres
    args.problem.radius_inner_bounding_sphere = 4000      # Radius of spherical risk-zone for collision with celestial body [m]
    args.problem.measurement_period = 2500 # Period for when a measurement sphere is recognized and managed. Unit: [seconds]

    # Create mesh of body.
    args.mesh.body, args.mesh.vertices, args.mesh.faces, args.mesh.largest_body_protuberant = create_mesh()

    # Initial position for integration (in cartesian coordinates):
    x = [-1.36986549e+03, -4.53113817e+03, -8.41816487e+03, -1.23505256e-01, -1.59791505e-01, 2.21471017e-01, 0, 0, 0, 0]
    x_osculating_elements = pk.ic2par(r=x[0:3], v=x[3:6], mu=args.body.mu) #translate to osculating orbital element

    # Compute trajectory via numerical integration as in UDP.
    _, list_of_ode_objects, _ = compute_trajectory(x_osculating_elements, args, compute_motion)

    # Get states along computed trajectory:
    positions, timesteps = get_trajectory_fixed_step(args, list_of_ode_objects)

    # Position and timesteps from previous working results (in cartesian coordinates):
    previous_positions = np.array([[-1369.86549, -1662.82224042, -1893.8561803, -2021.71832499, -2017.25938447, -1870.22440711, -1591.62823709, -1210.46048601, -766.70404768, -304.28302656, 133.43805324],
                          [-4531.13817, -4904.20028018, -5223.65538797, -5492.7850583, -5719.52297594, -5913.54217869, -6082.63727623, -6230.10562652, -6354.41963142, -6451.01289025, -6514.56664014],
                          [-8418.16487, -7815.6605588, -7126.20143363, -6378.67721074, -5617.45877143, -4896.55819511, -4270.49836708, -3785.17218501, -3471.87383633, -3345.51777456, -3405.6781468 ]])
    previous_timesteps = np.array([0, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000])
    
    assert all(np.isclose(previous_positions[0,:],positions[0,0:11],rtol=1e-5, atol=1e-5))
    assert all(np.isclose(previous_positions[1,:],positions[1,0:11],rtol=1e-5, atol=1e-5))
    assert all(np.isclose(previous_positions[2,:],positions[2,0:11],rtol=1e-5, atol=1e-5))
    assert all(np.isclose(previous_timesteps,timesteps[0:11],rtol=1e-5, atol=1e-5))

    # Assert steps in timesteps remain fixed.
    timesteps_diff = []
    for i in range(0,len(timesteps)-1):
        timesteps_diff.append(timesteps[i+1]-timesteps[i])
    assert all(x==timesteps_diff[0] for x in timesteps_diff)