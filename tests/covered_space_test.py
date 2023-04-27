
from toss import compute_space_coverage
import numpy as np
from ai.cs import sp2cart

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

    assert (ratio >= 0)


def test_perfect_ratio():
    """
    In this test we systematically test every point and see of we get a ration of 1.0.
    """
    
    # Defining dt=1
    timesteps = [0, 1]

    # Set radius boundaries:
    radius_min = 2
    radius_max = 8

    # Fixed maximal velocity from previously defined trajectory. 
    scaling_factor = 2
    velocities = np.array([-0.02826052, 0.1784372, -0.29885126]) * scaling_factor
    
    # Define frequency of points on grid:
    #max_velocity = np.sqrt(np.max(velocities[0,:]**2 + velocities[1,:]**2 + velocities[2,:]**2))
    max_velocity = np.max(np.linalg.norm(velocities))
    time_step = timesteps[1] - timesteps[0]
    max_distance_traveled = max_velocity * time_step

    # Calculate and adjust grid spacing based on maximal velocity and time step
    r_steps = np.floor((radius_max-radius_min)/max_distance_traveled)
    theta_steps = np.floor(np.pi*radius_min / max_distance_traveled)
    phi_steps = np.floor(2*np.pi*radius_min / max_distance_traveled)

    r = np.linspace(radius_min, radius_max, int(r_steps)) # Number of evenly spaced points along the radial axis
    theta = np.linspace(-np.pi/2, np.pi/2, int(theta_steps)) # Number of evenly spaced points along the polar angle/elevation (defined on [-pi/2, pi/2])
    phi = np.linspace(-np.pi, np.pi, int(phi_steps)) # Number of evenly spaced points along the azimuthal angle (defined on [-pi, pi])
    
    # Generate spherical coordinates for every point on the grid
    array_of_spherical_coordinates = np.empty((3, int(1e8)), dtype=np.float64)
    idx = 0
    for r_val in r:
        for theta_val in theta:
            for phi_val in phi:
                array_of_spherical_coordinates[:, idx] = [r_val, theta_val, phi_val]
                idx += 1   
                
    array_of_spherical_coordinates = array_of_spherical_coordinates[:, 0:idx]

    # Evaluate the ration of visited points, where the positions are every point on the corresponding grid.
    x_pos, y_pos, z_pos = sp2cart(array_of_spherical_coordinates[0,:], array_of_spherical_coordinates[1,:], array_of_spherical_coordinates[2,:])
    noise = 1e-2
    positions = np.vstack((x_pos,y_pos,z_pos)) - noise
    ratio = compute_space_coverage(positions, velocities, timesteps, radius_min, radius_max)

    assert (ratio==float(1))