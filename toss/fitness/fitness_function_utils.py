# Core packages
import numpy as np
from typing import Union

from toss.trajectory.equations_of_motion import rotate_point


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


def compute_space_coverage(number_of_spacecrafts: int, spin_axis: np.ndarray, spin_velocity: float, positions: np.ndarray, velocities: np.ndarray, timesteps: np.ndarray, radius_min: float, radius_max: float, max_velocity_scaling_factor: float) -> float:
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
        number_of_spacecrafts (int): Number of spacecraft
        spin_axis (np.ndarray): The axis around which the body rotates.
        spin_velocity (float): Angular velocity of the body's rotation.
        positions (np.ndarray): (3,N) Array of positions (in cartesian coordinates).
        velocities (np.ndarray): (3,N) Array of positions (in cartesian frame).
        timesteps (np.ndarray): (N) Array of time values for each position.
        radius_min (float): Inner radius of spherical grid, typically radius_inner_bounding_sphere.
        radius_max (float): Outer radius of spherical grid, typically radius_outer_bounding_sphere.
        max_velocity_scaling_factor (float): Scales the magnitude of the fixed-valued maximal velocity and therefore also the grid spacing.

    Returns:
        ratio (float): Number of True values to the total number of values in the boolean array
    """
    # Rotate positions according to body's rotation to simulate that the grid (i.e gravitational field approximation) is also rotating accrdingly
    rotated_positions = None
    
    pos = np.array_split(positions, number_of_spacecrafts, axis=1)
    for counter, pos_arr in enumerate(pos):

        rot_pos_arr = np.empty((pos_arr.shape))

        for col in range(0,len(pos_arr[0,:])):
            rot_pos_arr[:,col] = rotate_point(timesteps[col], pos_arr[:,col], spin_axis, spin_velocity)

        if counter == 0:
            rotated_positions = rot_pos_arr
        else:
            rotated_positions = np.hstack((rotated_positions, rot_pos_arr))


    # Fixed maximal velocity from previously defined trajectory. 
    # We use a fixed value to avoid prioritizing higher velocities. 
    #
    # NOTE: The fitness value for covered volume will not be feasible if
    #        its maximal velocity exceeds the provided value below as the grid
    #        spacing will be too small. Please use the scaling factor to adapt for this.
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

    # Convert the positions along the trajectory to spherical coordinates
    r_points, theta_points, phi_points = cart2sphere(rotated_positions[0,:], rotated_positions[1,:], rotated_positions[2,:])

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

        # Define a zero-valued tensor with the same shape as the spherical meshgrid
        r_idx = np.where(bool_tensor == True)[0]
        sum_of_weights = np.sum(1/r[r_idx])

        # Return fitness
        fitness = ratio + sum_of_weights

        return fitness


def get_spherical_tensor_grid(timesteps: np.ndarray, radius_min: float, radius_max: float, max_velocity_scaling_factor: float) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a number of points in each spherical axis, which together defines
    a spherical tensor grid that satisfies the Courant–Friedrichs–Lewy condition.
    The function also generates a corresponding boolean tensor initialized by False-values
    indicating none of the defined points have been visited by a spacecraft.

    Args:
        timesteps (np.ndarray): (N) Array of time values for each position.
        radius_min (float): Inner radius of spherical grid, typically radius_inner_bounding_sphere.
        radius_max (float): Outer radius of spherical grid, typically radius_outer_bounding_sphere.
        max_velocity_scaling_factor (float): Scales the magnitude of the fixed-valued maximal velocity and therefore also the grid spacing.

    Returns:
        r (np.ndarray): Array of r coordinates for each point defined on the spherical tensor.
        theta (np.ndarray): Array of theta coordinates for each point defined on the spherical tensor.
        phi (np.ndarray): Array of phi coordinates for each point defined on the spherical tensor.
        bool_tensor (np.ndarray): Boolean array corresponding to each point defined on the spherical tensor.
    """

    # Fixed maximal velocity from previously defined trajectory. 
    # We use a fixed value to avoid prioritizing higher velocities. 
    #
    # NOTE: The fitness value for covered volume will not be feasible if
    #        its maximal velocity exceeds the provided value below as the grid
    #        spacing will be too small. Please use the scaling factor to adapt for this.
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

    # Create a boolean tensor with the same shape as the spherical meshgrid
    bool_tensor = np.full((len(r), len(theta), len(phi)), False)

    return r, theta, phi, bool_tensor


def cart2sphere(x, y, z) -> tuple:
    """
    Converts array of cartesian coordinates to corresponding spherical coordinates.
    The elevation/polar angle theta is defined from the x-y plane up.
    The azimuthal angle phi is defined on the x-y plane. 

    NOTE: in mind:
    - arcsin is defined on [-pi, pi]
    - arctan2 is defined on [-pi/2, pi/2]

    Args:
        x (scalar or array_like): X-component of data.
        y (scalar or array_like): Y-component of data.
        z (scalar or array_like): Z-component of data.

    Returns:
        tuple (r, theta, phi) of data in spherical coordinates.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    scalar_input = False
    if x.ndim == 0 and y.ndim == 0 and z.ndim == 0:
        x = x[None]
        y = y[None]
        z = z[None]
        scalar_input = True
    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arctan2(z,np.sqrt(x**2 + y**2))
    phi = np.arctan2(y, x)
    if scalar_input:
        return (r.squeeze(), theta.squeeze(), phi.squeeze())
    return (r, theta, phi)

def sphere2cart(r, theta, phi) -> tuple:
    """
    Converts array of spherical coordinates to corresponding cartesian coordinates.

    Args:
        r (scalar or array_like): R-component of data.
        theta (scalar or array_like): Theta-component of data.
        phi (scalar or array_like): Phi-component of data.

    Returns:
        tuple (x, y, z) of data in cartesian coordinates.
    """
    r = np.asarray(r)
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    scalar_input = False
    if r.ndim == 0 and theta.ndim == 0 and phi.ndim == 0:
        r = r[None]
        theta = theta[None]
        phi = phi[None]
        scalar_input = True
    x = r*np.cos(theta)*np.cos(phi)
    y = r*np.cos(theta)*np.sin(phi)
    z = r*np.sin(theta)
    if scalar_input:
        return (x.squeeze(), y.squeeze(), z.squeeze())
    return (x, y, z)
