""" This test checks whether or not the integration is performed correctly """

# Import required modules
from toss import compute_motion
from toss import create_mesh
from toss import compute_trajectory
from toss import get_trajectory_fixed_step
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
    positions, _, timesteps = get_trajectory_fixed_step(args, list_of_ode_objects)

    # Position and timesteps from previous working results (in cartesian coordinates):
    previous_positions = np.array([
        [-135.13402075, 278.25186467, 713.90842453, 1185.91569478, 1704.379756, 2277.13445551, 2911.01746536, 3612.55429757, 4387.57252959, 5239.18486301, 6164.10258861, 7148.9059736, 8169.51784163, 9195.66312775, 10197.91204589, 11153.57923919, 12049.78936608, 12883.80222618, 13661.46107406, 14394.65936702, 15098.68426644, 15789.68558432, 16482.46633303, 17188.84949224, 17916.72182947, 18669.80615553, 19448.02594818, 20248.24598722, 21065.22666557],
        [-4089.53592604, -4582.87500723, -4981.34761159, -5287.71233958, -5501.8823638, -5622.04908503, -5645.32203174, -5568.06503901, -5386.19059618, -5095.81271711, -4694.68940867, -4184.39742931, -3572.17995868, -2871.09210085, -2098.40293603, -1273.35071283, -415.03797937, 459.26857293, 1335.31001252, 2202.38940261, 3053.48768832, 3885.05394287, 4696.55977473, 5489.79625194, 6268.08376158, 7035.46345184, 7795.99959288, 8553.24764699, 9309.90452067],
        [6050.17636635, 6396.86611796, 6601.81120629, 6686.18379911, 6666.93701688, 6558.37640111, 6373.57312606, 6126.09146152, 5832.06103843, 5512.04735559, 5191.28625259, 4896.52663523, 4649.65435439, 4461.68786725, 4330.89102597, 4245.09742664, 4186.08867632, 4134.10036372, 4071.40078178, 3984.5349734, 3865.41893011, 3711.5076439, 3525.24663912, 3312.87678088, 3082.97864716, 2844.95547407, 2607.72921137, 2378.76549785, 2163.4641028]])
    previous_timesteps = np.array([0, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000, 32500, 35000, 37500, 40000, 42500, 45000, 47500, 50000, 52500, 55000, 57500, 60000, 62500, 65000, 67500, 70000,])
    
    assert all(np.isclose(previous_positions[0,:],positions[0,0:],rtol=1e-5, atol=1e-5))
    assert all(np.isclose(previous_positions[1,:],positions[1,0:],rtol=1e-5, atol=1e-5))
    assert all(np.isclose(previous_positions[2,:],positions[2,0:],rtol=1e-5, atol=1e-5))
    assert all(np.isclose(previous_timesteps,timesteps,rtol=1e-5, atol=1e-5))

    # Assert steps in timesteps remain fixed.
    timesteps_diff = []
    for i in range(0,len(timesteps)-1):
        timesteps_diff.append(timesteps[i+1]-timesteps[i])
    assert all(x==timesteps_diff[0] for x in timesteps_diff)
