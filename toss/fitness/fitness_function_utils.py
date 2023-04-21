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


def compute_space_coverage(positions, radius_outer_bounding_sphere):
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
        positions (_type_): (3,N) Array of positions (in cartesian coordinates).
        radius_outer_bounding_sphere (float): Radius of outer bounding sphere.

    Returns:
        ratio (float): Number of True values to the total number of values in the boolean array
    """
    # Set the detail of the spherical meshgrid:
    r_num = 1000 # Number of evenly spaced points along the radial axis
    theta_num = 1450 # Number of evenly spaced points along the polar angle (defined in between 0 and pi)
    phi_num = 1450 # Number of evenly spaced points along the azimuthal angle (defined in between 0 and 2*pi)

    # Create a spherical meshgrid
    r = np.linspace(0, radius_outer_bounding_sphere, r_num) # radial coordinates
    theta = np.linspace(0, np.pi, theta_num) # polar angle coordinates
    phi = np.linspace(0, 2*np.pi, phi_num) # azimuthal angle coordinates
    r_matrix, theta_matrix, phi_matrix = np.meshgrid(r, theta, phi) # 3D arrays of r, theta and phi
    
    # Create a boolean array with the same shape as the spherical meshgrid
    bool_array = np.zeros_like(r_matrix, dtype=bool) # initialize with False

    # Create grid of positions
    x = positions[0,:]
    y = positions[1,:]
    z = positions[2,:]
    x2, y2, z2 = np.meshgrid(x, y, z, indexing='ij')
    all_grid = np.array([x2.flatten(), y2.flatten(), z2.flatten()]).T

    # Remove points that are defined outside the outer bounding sphere
    point_radius = np.linalg.norm(all_grid, axis=1)
    in_points = point_radius < radius_outer_bounding_sphere
    points = all_grid[in_points]

    # For each point in the set, find its spherical coordinates and indices in the meshgrid
    for point in points:
        x, y, z = point # cartesian coordinates
        r_point = np.sqrt(x**2 + y**2 + z**2) # radial coordinate
        theta_point = np.arccos(z / r_point) # polar angle coordinate
        phi_point = np.arctan2(y, x) # azimuthal angle coordinate
        i = np.argmin(np.abs(r - r_point)) # index of closest r value in r_matrix
        j = np.argmin(np.abs(theta - theta_point)) # index of closest theta value in theta_matrix
        k = np.argmin(np.abs(phi - phi_point)) # index of closest phi value in phi_matrix
        bool_array[j, i, k] = True # set the value to True where the point is located

    # Compute the ratio of True values to the total number of values in the boolean array
    ratio = bool_array.sum() / bool_array.size

    return ratio