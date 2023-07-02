from toss import compute_space_coverage, sphere2cart, setup_parameters, rotate_point
import numpy as np

def test_large_random_sample():
    """
    In this test we sample 1 million positions to evaluate if the code can manage such quantity of positions.
    """

    # Setup parameters according to default cfg. 
    args = setup_parameters()
    number_of_spacecrafts = 1

    # Number of positions to be considered for evaluation
    number_of_samples = 10000000

    # Set radius boundaries:
    radius_min = 4
    radius_max = 10

    # Define a list of times corresponding to each position
    timesteps = np.arange(0, number_of_samples+1, 1)

    # Generate random sample of points defined on [radius_min, radius_max]
    positions = (radius_max)*np.random.random_sample((3,number_of_samples)) + (radius_min/2)

    # Generate random sample of velocities defined on [-1, 1]
    velocities = 2*np.random.random_sample((3,number_of_samples)) - 1

    # Compute ratio of visited points on the spherical meshgrid
    max_velocity_scaling_factor = 1
    coverage = compute_space_coverage(number_of_spacecrafts, args.body.spin_axis, args.body.spin_velocity, positions, velocities, timesteps, radius_min, radius_max, args.problem.tensor_grid_r, args.problem.tensor_grid_theta, args.problem.tensor_grid_phi, args.problem.bool_tensor)

    assert (coverage >= 0)


def test_perfect_ratio():
    """
    In this test we systematically test every point and see of we get a coverage of >1 (Recall: Coverage = ra†io + weights).
    """

    # Setup parameters according to default cfg. 
    args = setup_parameters()
    number_of_spacecrafts = 1
    
    # Define a list of times corresponding to each position
    number_of_samples = 1000000 # choose large number of samples to guarantee that we cover all positions. 
    timesteps = np.arange(0, number_of_samples+1, 1)

    # Set radius boundaries:
    radius_min = 2
    radius_max = 8

    # Fixed maximal velocity from previously defined trajectory. 
    max_velocity_scaling_factor = 1
    fixed_velocity = np.array([-0.02826052, 0.1784372, -0.29885126])

    # Define frequency of points for the spherical meshgrid: (see: Courant–Friedrichs–Lewy condition)
    max_velocity = np.max(np.linalg.norm(fixed_velocity)) * max_velocity_scaling_factor
    time_step = timesteps[1]-timesteps[0]
    max_distance_traveled = max_velocity * time_step

    # Calculate and adjust grid spacing based on maximal velocity and time step
    r_steps = np.floor((radius_max-radius_min)/max_distance_traveled)
    theta_steps = np.floor(np.pi*radius_min / max_distance_traveled)
    phi_steps = np.floor(2*np.pi*radius_min / max_distance_traveled)

    r = np.linspace(radius_min, radius_max, int(r_steps)) # Number of evenly spaced points along the radial axis
    theta = np.linspace(-np.pi/2, np.pi/2, int(theta_steps)) # Number of evenly spaced points along the polar angle/elevation (defined on [-pi/2, pi/2])
    phi = np.linspace(-np.pi, np.pi, int(phi_steps)) # Number of evenly spaced points along the azimuthal angle (defined on [-pi, pi])

    # Addition of noise to avoid approximation errors for max-min points
    noise = 1e-4
    r[0] += noise
    r[-1] -= noise
    theta[0] += noise
    theta[-1] -= noise
    phi[0] += noise
    phi[-1] -= noise

    # Generate spherical coordinates for every point on the grid
    array_of_spherical_coordinates = np.empty((3, int(1e8)), dtype=np.float64)
    idx = 0
    for r_val in r:
        for theta_val in theta:
            for phi_val in phi:
                array_of_spherical_coordinates[:, idx] = [r_val, theta_val, phi_val]
                idx += 1   
                
    array_of_spherical_coordinates = array_of_spherical_coordinates[:, 0:idx]

    # Translate positions from spherical to cartesian coordinate 
    x_pos, y_pos, z_pos = sphere2cart(array_of_spherical_coordinates[0,:], array_of_spherical_coordinates[1,:], array_of_spherical_coordinates[2,:])
    positions = np.vstack((x_pos,y_pos,z_pos))

    # Rotate each position backwards to counteract the rotation made to simulate a rotating gravity field when computing coverage. 
    rotated_positions = None
    pos = np.array_split(positions, number_of_spacecrafts, axis=1)
    for counter, pos_arr in enumerate(pos):

        rot_pos_arr = np.empty((pos_arr.shape))

        for col in range(0,len(pos_arr[0,:])):
            rot_pos_arr[:,col] = rotate_point(-timesteps[col], pos_arr[:,col], args.body.spin_axis, args.body.spin_velocity)

        if counter == 0:
            rotated_positions = rot_pos_arr
        else:
            rotated_positions = np.hstack((rotated_positions, rot_pos_arr))

    # Evaluate the coverage of visited points, where the positions are every point on the corresponding grid.
    coverage = compute_space_coverage(number_of_spacecrafts, args.body.spin_axis, args.body.spin_velocity, positions, fixed_velocity, timesteps, radius_min, radius_max, args.problem.tensor_grid_r, args.problem.tensor_grid_theta, args.problem.tensor_grid_phi, args.problem.bool_tensor)

    assert (coverage >= 1)
