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
        [-135.13402075, 263.34247213, 604.83605835, 849.51872286, 975.35622434, 975.37761256, 856.59771949, 638.45266394, 350.18132935, 27.1853693, -294.28853045, -583.90759178, -822.3913349, -1002.11574267, -1124.45517542, -1196.93379905, -1231.33132683, -1242.11083363, -1244.49663412, -1252.35205866, -1276.50765489, -1323.95737531, -1397.72196179, -1497.17677682, -1618.6327649, -1756.12575258, -1902.37604852, -2049.82067983, -2191.61684185],
        [-4089.53592604, -4582.39401788, -4986.05753223, -5326.50900958, -5637.29491187, -5954.91401946, -6314.83709303, -6747.45090633, -7274.51213072, -7906.14772622, -8638.85442205, -9456.38779996, -10334.6839721, -11248.15478543, -12173.77309256, -13092.54391164, -13989.99966257, -14856.75042157, -15688.9041651, -16487.78801298, -17258.82363434, -18010.00195367, -18750.44089875, -19489.15603228, -20234.06615471, -20991.21820906, -21764.27683641, -22554.33884122, -23360.05030541],
        [6050.17636635, 6398.22518084, 6615.11263878, 6735.57340647, 6789.4650926, 6802.2702936, 6795.40677054, 6785.75530166, 6785.00389449, 6799.10003387, 6828.10262589, 6867.1215357, 6908.57236613, 6944.63010631, 6968.71867279, 6976.09892661, 6964.16545987, 6932.68012285, 6883.75078312, 6821.38333952, 6750.70954085, 6677.18925678, 6605.96647813, 6541.39671286, 6486.71468875, 6443.83011983, 6413.2677687, 6394.26514038, 6385.00412717]])
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