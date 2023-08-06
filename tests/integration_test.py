""" This test checks whether or not the integration is performed correctly """

# Import required modules
from toss import compute_motion
from toss import create_mesh
from toss import compute_trajectory
from toss import get_trajectory_adaptive_step
from toss import setup_parameters

# Core packages
import numpy as np

def test_integration():
    # Load parameters from default cfg.
    args = setup_parameters()

    # Adjust for test-specific parameters:
    args.problem.start_time = 0          
    args.problem.final_time = 20*3600.0   
    args.problem.initial_time_step = 600  
    args.problem.activate_event = True   
    args.problem.number_of_maneuvers = 0 
    args.problem.activate_rotation = True
    args.problem.radius_inner_bounding_sphere = 4000 
    args.problem.measurement_period = 2500 
    args.mesh.mesh_path = "3dmeshes/churyumov-gerasimenko_llp.pk"
    args.mesh.body, args.mesh.vertices, args.mesh.faces, args.mesh.largest_body_protuberant = create_mesh(args.mesh.mesh_path)

    # Initial position for integration (in cartesian coordinates):
    #   NOTE: The initial state vector is structured as x = [rx, ry, rz, v_magnitude, vx, vy, vz]
    #         And represents the optimal initial position found for the single spacecraft case 
    #         presented in: https://doi.org/10.48550/arXiv.2306.01602. 
    x = np.array([-135.13402075, -4089.53592604, 6050.17636635, 2.346971623591584122e-01, 6.959989121956766667e-01, -9.249848356174805719e-01, 7.262727928440093628e-01])

    # Compute trajectory via numerical integration as in UDP.
    _, list_of_ode_objects, _ = compute_trajectory(x, args, compute_motion)

    # Get states along computed trajectory:
    states_new, _ = get_trajectory_adaptive_step(list_of_ode_objects)

    # Final state from previous working results (in cartesian coordinates):
    final_state_historical = [2.55718161e+04, -2.28183531e+04, 2.06704540e+03, 3.87911600e-01, -3.01262020e-01, -7.47139451e-02]

    # New final state:
    final_state_new = states_new[0:6,-1]
    assert all(np.isclose(final_state_historical,final_state_new,rtol=1e-5, atol=1e-5))