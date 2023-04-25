# Core packages
import numpy as np
import typing

def estimate_covered_volume(positions: np.ndarray) -> float:
    """Estimates the volume covered by the trajectory through spheres around sampling points.

    Args:
        positions (np.ndarray): (3,N) array containing information given satellite positions at given times (expressed in cartesian frame as (x,y,z)).
    
    Returns:
        sphere_radii (np.ndarray): (1,N) array of radii for each measurement sphere corresponding to a given position.
        estimated_volume (float): Estimated volume covered by the trajectory.
    """

    # Init array to hold spheres radius around sampling points to exactly cover 
    # the distance between two consecutive points:
    sphere_radii = np.empty((len(positions[0,:])), dtype=np.float64)

    # Compute the distance between consecutive points
    distances_between_positions = np.linalg.norm(positions[:,1:] - positions[:,:-1], axis=0)

    # Set radius to cover that distance, first and last point have the same radius
    # as the second and second to last point respectively:
    sphere_radii[1:] = distances_between_positions/2
    sphere_radii[0] = sphere_radii[1]
    sphere_radii[-1] = sphere_radii[-2]

    # Compute volume of each sphere
    sphere_volumes = (4/3) * np.pi * (sphere_radii**3)

    # Compute estimated volume covered by the trajectory
    estimated_volume = np.sum(sphere_volumes)
    
    return sphere_radii, estimated_volume


def _compute_squared_distance(positions,constant):
    """ Compute squared distance from each point in a given array to some given constant.

    Args:
        positions (np.ndarray): (3,N) Array of positions (in cartesian coordinates).
        constant (float): Constant value. 

    Returns:
        (np.ndarray): (N,) array containing average distance from each point in arr to constant.
    """
    return np.sum(np.power(positions,2), axis=0) - constant**2


def compute_space_coverage(positions, velocities, timesteps, radius_outer_bounding_sphere):
    """
    In this function, we create a spherical meshgrid by defining a number of points
    inside the outer bounding sphere. We then generate a multidimensional array
    of boolean values corresponding to each point in the meshgrid. These values 
    are initialy set to false since they have not been visited by any spacecraft.

    Given a set of posiions on the trajectory that are defined inside the 
    outer bounding sphere, we identify the points that are the closest the given positions, 
    and subsequently reassign the visited boolean value to true. The ratio of visited points 
    is then given by the number of True values to the total number of values in the boolean array.

    Args:
        positions (np.ndarray): (3,N) Array of positions (in cartesian coordinates).
        velocities (np.ndarray): (3,N) Array of positions (in cartesian frame).
        timesteps (np.ndarray): (N) Array of time values for each position.
        radius_outer_bounding_sphere (float): Radius of outer bounding sphere.

    Returns:
        ratio (float): Number of True values to the total number of values in the boolean array
    """

    # Define frequency of points for the spherical meshgrid: (see: Courant–Friedrichs–Lewy condition)
    cfl = 1 # Courant number
    v_max = np.sqrt(np.max(velocities[0,:]**2 + velocities[1,:]**2 + velocities[2,:]**2))
    dt = timesteps[1]-timesteps[0]
    
    # Calculate and adjust grid spacing based on the Courant number, maximal velocity, and time step
    grid_spacing = v_max * dt / cfl
    r = np.linspace(0, radius_outer_bounding_sphere, int(radius_outer_bounding_sphere / grid_spacing)) # Number of evenly spaced points along the radial axis
    theta = np.linspace(0, np.pi, int(np.pi / grid_spacing)) # Number of evenly spaced points along the polar angle (defined in between 0 and pi)
    phi = np.linspace(0, 2 * np.pi, int(2 * np.pi / grid_spacing)) # Number of evenly spaced points along the azimuthal angle (defined in between 0 and 2*pi)

    # Create a spherical meshgrid
    r_matrix, theta_matrix, phi_matrix = np.meshgrid(r, theta, phi)

    # Convert the positions along the trajectory to spherical coordinates
    r_points = np.sqrt(positions[0,:]**2 + positions[1,:]**2 + positions[2,:]**2) # radial coordinates of points
    theta_points = np.arccos(positions[2,:] / r_points) # polar angle coordinates of points
    phi_points = np.arctan2(positions[1,:], positions[0,:]) # azimuthal angle coordinates of points

    # Remove points outside measurement zone (i.e outside outer-bounding sphere)
    index_feasible_positions = r_points < radius_outer_bounding_sphere
    r_points = r_points[index_feasible_positions]
    theta_points = theta_points[index_feasible_positions]
    phi_points = phi_points[index_feasible_positions]

    # Find the indices of the closest values in the meshgrid for each point using broadcasting
    i = np.argmin(np.abs(r[:, np.newaxis] - r_points), axis=0) # indices along r axis
    j = np.argmin(np.abs(theta[:, np.newaxis] - theta_points), axis=0) # indices along theta axis
    k = np.argmin(np.abs(phi[:, np.newaxis] - phi_points), axis=0) # indices along phi axis

    # Create a boolean tensor with the same shape as the spherical meshgrid
    bool_array = np.zeros_like(r_matrix, dtype=bool) # initialize with False

    # Set the values to True where the points are located using advanced indexing
    bool_array[j, i, k] = True

    # Compute the ratio of True values to the total number of values in the boolean array
    ratio = bool_array.sum() / bool_array.size

    return ratio