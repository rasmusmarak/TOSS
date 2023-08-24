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
        [-135.13402075, 279.18588568, 722.3459321, 1216.10791375, 1777.11430242, 2416.89891887, 3141.99448684, 3953.67815967, 4847.08414504, 5809.82774266, 6821.45139681, 7855.80937286, 8886.65434476, 9893.13138873, 10862.25268005, 11788.65512963, 12673.18396574, 13521.25780487, 14341.32956517, 15143.49723331, 15938.16858981, 16734.7890487, 17540.82641509, 18361.19266825, 19198.18557898, 20051.7536288, 20919.90766019, 21799.22992697, 22685.44762944],
        [-4089.53592604, -4583.56118047, -4989.46104201, -5323.18841157, -5603.11343039, -5849.34346439, -6083.19740516, -6326.90477521, -6603.07017291, -6933.01378491, -7333.19647567, -7810.71739179, -8361.25287843, -8971.37271147, -9623.06098478, -10297.6253255, -10978.15257987, -11650.84455206, -12305.66868768, -12936.6376696, -13541.74458408, -14122.5870485, -14683.50918791, -15230.52227774, -15770.22099528, -16308.92933565, -16852.07203161, -17403.73084201, -17966.36495188],
        [6050.17636635, 6396.6506026, 6599.62043763, 6677.62106049, 6644.6052458, 6511.76615511, 6288.69847552, 5984.28838134, 5607.5037986, 5168.1142464, 4677.09536147, 4146.1693177, 3586.41305902, 3006.92953876, 2414.44722537, 1813.67286854, 1207.85409486, 599.25383879, -10.51829532, -620.2964709, -1229.22891747, -1836.7053999, -2442.30326138, -3045.74240953, -3646.84171646, -4245.4852347, -4841.60177923, -5435.15628437, -6026.14766051]])
    previous_timesteps = np.array([0, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000, 32500, 35000, 37500, 40000, 42500, 45000, 47500, 50000, 52500, 55000, 57500, 60000, 62500, 65000, 67500, 70000])

    assert all(np.isclose(previous_positions[0,:],positions[0,0:],rtol=1e-5, atol=1e-5))
    assert all(np.isclose(previous_positions[1,:],positions[1,0:],rtol=1e-5, atol=1e-5))
    assert all(np.isclose(previous_positions[2,:],positions[2,0:],rtol=1e-5, atol=1e-5))
    assert all(np.isclose(previous_timesteps,timesteps,rtol=1e-5, atol=1e-5))

    # Assert steps in timesteps remain fixed.
    timesteps_diff = []
    for i in range(0,len(timesteps)-1):
        timesteps_diff.append(timesteps[i+1]-timesteps[i])
    assert all(x==timesteps_diff[0] for x in timesteps_diff)