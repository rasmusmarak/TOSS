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
        [-135.13402075, 282.1773485, 743.0410202, 1276.28224613, 1899.66020555, 2621.96145056, 3444.16827025, 4359.49804523, 5353.15908791, 6403.47129378, 7485.36289758, 8575.04431288, 9653.49391347, 10707.7804709, 11730.84142377, 12720.63287221, 13679.07660813, 14611.00304399, 15523.13418576, 16423.12007258, 17318.628973, 18216.50490245, 19122.11847268, 20039.01401806, 20968.84492828, 21911.51893139, 22865.48010767, 23828.07664004, 24795.99796002],
        [-4089.53592604, -4585.29216079, -5003.51280058, -5370.20809984, -5711.77858241, -6053.80802937, -6420.40856041, -6833.2610143, -7309.61144947, -7859.53859028, -8484.18516104, -9176.5119778, -9924.07245895, -10712.05688924, -11525.54667526, -12350.92711804, -13176.72665702, -13994.08429256, -14796.96057238, -15582.13981475, -16349.05600627, -17099.41813922, -17836.6000544, -18564.93615054, -19289.05163706, -20013.30553611, -20741.36406314, -21475.89618336, -22218.39341265],
        [6050.17636635, 6396.29093147, 6598.7869624, 6680.4465535, 6662.21971671, 6563.76783638, 6404.13960796, 6202.30661104, 5976.87150259, 5744.55365609, 5518.10501398, 5305.08313761, 5108.17216298, 4926.46827073, 4756.88553741, 4595.28051436, 4437.27382671, 4278.82130686, 4116.60363198, 3948.27118839, 3772.56975042, 3589.34382525, 3399.38980828, 3204.21008005, 3005.73046585, 2806.03004452, 2607.10357284, 2410.66199796, 2217.97638338]])
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