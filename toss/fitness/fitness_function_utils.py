# Core packages
import numpy as np
import typing
from ai.cs import cart2sp, sp2cart

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


def _compute_squared_distance(positions: np.ndarray, constant: float) -> np.ndarray:
    """ Compute squared distance from each point in a given array to some given constant.

    Args:
        positions (np.ndarray): (3,N) Array of positions (in cartesian coordinates).
        constant (float): Constant value. 

    Returns:
        (np.ndarray): (N,) array containing average distance from each point in arr to constant.
    """
    return np.sum(np.power(positions,2), axis=0) - constant**2


def compute_space_coverage(positions: np.ndarray, velocities: np.ndarray, timesteps: np.ndarray, radius_min: float, radius_max: float) -> float:
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
        radius_min (float): Inner radius of spherical grid, typically radius_inner_bounding_sphere.
        radius_max (float): Outer radius of spherical grid, typically radius_outer_bounding_sphere.

    Returns:
        ratio (float): Number of True values to the total number of values in the boolean array
    """

    # Fixed maximal velocity from previously defined trajectory. 
    # We use a fixed value to avoid prioritizing higher velocities. 
    #
    # NOTE: The fitness value for covered volume will not be feasible if
    #        its maximal velocity exceeds the provided value below as the grid
    #        spacing will be too small. Please use the scaling factor to adapt for this.
    scaling_factor = 1
    fixed_velocity = np.array([-0.02826052, 0.1784372, -0.29885126]) * scaling_factor

    # Define frequency of points for the spherical meshgrid: (see: Courant–Friedrichs–Lewy condition)
    max_velocity = np.max(np.linalg.norm(fixed_velocity))
    time_step = timesteps[1]-timesteps[0]
    max_distance_traveled = max_velocity * time_step

    # Calculate and adjust grid spacing based on maximal velocity and time step
    r_steps = np.floor((radius_max-radius_min)/max_distance_traveled)
    theta_steps = np.floor(np.pi*radius_min / max_distance_traveled)
    phi_steps = np.floor(2*np.pi*radius_min / max_distance_traveled)

    r = np.linspace(radius_min, radius_max, int(r_steps)) # Number of evenly spaced points along the radial axis
    theta = np.linspace(-np.pi/2, np.pi/2, int(theta_steps)) # Number of evenly spaced points along the polar angle/elevation (defined on [-pi/2, pi/2])
    phi = np.linspace(-np.pi, np.pi, int(phi_steps)) # Number of evenly spaced points along the azimuthal angle (defined on [-pi, pi])

    # Convert the positions along the trajectory to spherical coordinates
    r_points, theta_points, phi_points = cart2sp(positions[0,:], positions[1,:], positions[2,:])

    # Remove points outside measurement zone (i.e outside outer-bounding sphere)
    index_feasible_positions = np.where(r_points <= radius_max) # Addition of noise to cover approximation error
    r_points = r_points[index_feasible_positions]
    theta_points = theta_points[index_feasible_positions]
    phi_points = phi_points[index_feasible_positions]

    # Remove points inside safety-radius (i.e inside inner-bounding sphere)
    index_feasible_positions = np.where(r_points >= radius_min) # Addition of noise to cover approximation error
    r_points = r_points[index_feasible_positions]
    theta_points = theta_points[index_feasible_positions]
    phi_points = phi_points[index_feasible_positions]

    # Compute ratio of visited points. 
    if len(r_points)==0 or len(theta_points)==0 or len(phi_points)==0:
        ratio = 0
        return ratio
    
    else: 
        # Find the indices of the closest values in the meshgrid for each point using broadcasting
        i = np.argmin(np.abs(r[:, np.newaxis] - r_points), axis=0) # indices along r axis
        j = np.argmin(np.abs(theta[:, np.newaxis] - theta_points), axis=0) # indices along theta axis
        k = np.argmin(np.abs(phi[:, np.newaxis] - phi_points), axis=0) # indices along phi axis

        # Create a boolean tensor with the same shape as the spherical meshgrid
        bool_tensor = np.full((len(r), len(theta), len(phi)), False)

        # Set the values to True where the points are located using advanced indexing
        bool_tensor[i, j, k] = True

        # Compute the ratio of True values to the total number of values in the boolean tensor
        ratio = bool_tensor.sum() / bool_tensor.size

        return ratio