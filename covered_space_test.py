
from toss import compute_space_coverage
import numpy as np
from ai.cs import cart2sp, sp2cart

def test_large_random_sample():
    """
    In this test we sample 1 million positions to evaluate if the code can manage such quantity of positions.
    """

    number_of_samples = 10000000

    # Defining dt=1
    timesteps = [0, 1]

    # Set radius boundaries:
    radius_min = 1
    radius_max = 10

    # Generate random sample of points defined on [radius_min, radius_max]
    positions = (radius_max-radius_min)*np.random.random_sample((3,number_of_samples)) + radius_min

    print(positions)

    # Generate random sample of velocities defined on [-1, 1]
    velocities = 2*np.random.random_sample((3,number_of_samples)) - 1

    # Compute ration of visited points on the spherical meshgrid
    ratio = compute_space_coverage(positions, velocities, timesteps, radius_min, radius_max)

    print(ratio)

    assert (ratio >= 0)


def test_perfect_ratio():
    """
    In this test we systematically test every point and see of we get a ration of 1.0.
    """

    number_of_samples = 1000

    # Defining dt=1
    timesteps = [0, 1]

    # Set radius boundaries:
    radius_min = 2
    radius_max = 8

    # Generate random sample of velocities defined on [-1, 1]
    velocities = 2*np.random.random_sample((3,number_of_samples)) - 1
    
    # Define frequency of points on grid:
    max_velocity = np.sqrt(np.max(velocities[0,:]**2 + velocities[1,:]**2 + velocities[2,:]**2))
    time_step = timesteps[1] - timesteps[0]
    max_distance_traveled = max_velocity * time_step

    # Calculate and adjust grid spacing based on maximal velocity and time step
    r_steps = np.floor((radius_max-radius_min)/max_distance_traveled)
    theta_steps = np.floor(np.pi*radius_min / max_distance_traveled)
    phi_steps = np.floor(2*np.pi*radius_min / max_distance_traveled)

    r = np.linspace(radius_min, radius_max, int(r_steps)) # Number of evenly spaced points along the radial axis
    #theta = np.linspace(0, np.pi, int(theta_steps)) # Number of evenly spaced points along the polar angle (defined in between 0 and pi)
    #phi = np.linspace(0, 2 * np.pi, int(phi_steps)) # Number of evenly spaced points along the azimuthal angle (defined in between 0 and 2*pi)
    theta = np.linspace(-np.pi/2, np.pi/2, int(theta_steps))
    phi = np.linspace(-np.pi, np.pi, int(phi_steps))


    ########################
    
    # Generate spherical coordinates for every point on the grid
    array_of_spherical_coordinates = np.empty((3, int(1e8)), dtype=np.float64)
    idx = 0
    for r_val in r:
        for theta_val in theta:
            for phi_val in phi:
                array_of_spherical_coordinates[:, idx] = [r_val, theta_val, phi_val]
                idx += 1   
                
    array_of_spherical_coordinates = array_of_spherical_coordinates[:, 0:idx]
    
    # Convert the spherical coordinates to cartesian coordinates
    #array_of_cartesian_coordinates = np.empty((array_of_spherical_coordinates.shape), dtype=np.float64)
    #array_of_cartesian_coordinates[0,:] = array_of_spherical_coordinates[0,:] * np.sin(array_of_spherical_coordinates[1,:]) * np.cos(array_of_spherical_coordinates[2,:])
    #array_of_cartesian_coordinates[1,:] = array_of_spherical_coordinates[0,:] * np.sin(array_of_spherical_coordinates[1,:]) * np.sin(array_of_spherical_coordinates[2,:])
    #array_of_cartesian_coordinates[2,:] = array_of_spherical_coordinates[0,:] * np.cos(array_of_spherical_coordinates[1,:])     

    # Evaluate the ration of visited points, where the positions are every point on the corresponding grid.
    #positions = array_of_cartesian_coordinates
    x_pos, y_pos, z_pos = sp2cart(array_of_spherical_coordinates[0,:], array_of_spherical_coordinates[1,:], array_of_spherical_coordinates[2,:])
    positions = np.vstack((x_pos,y_pos,z_pos))
    ratio = compute_space_coverage(positions, velocities, timesteps, radius_min, radius_max)


    ########################



    # Create a spherical meshgrid
    #r_matrix, theta_matrix, phi_matrix = np.meshgrid(r, theta, phi)

    # Create a boolean tensor with the same shape as the spherical meshgrid
    #bool_array = np.zeros_like(r_matrix, dtype=bool) # initialize with False

    # Systematically go over every possible point on the meshgrid:
    #for i, r_val in enumerate(r):
    #    for j, theta_val in enumerate(theta):
    #        for k, phi_val in enumerate(phi):
    #            bool_array[j, i, k] = True

    # Compute the ratio of True values to the total number of values in the boolean array
    #ratio = bool_array.sum() / bool_array.size

    assert (ratio==float(1))