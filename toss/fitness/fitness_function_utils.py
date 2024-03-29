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


def compute_space_coverage(number_of_spacecrafts: int, spin_axis: np.ndarray, spin_velocity: float, positions: np.ndarray, velocities: np.ndarray, timesteps: np.ndarray, radius_min: float, radius_max: float, r: np.ndarray, theta: np.ndarray, phi: np.ndarray, bool_tensor: np.ndarray) -> float:
    """
    Given a set of posiions on the trajectory that are defined inside the 
    outer bounding sphere, we identify the points that are the closest the given positions, 
    and subsequently compute the ratio of visited points by the number of True values to the 
    total number of subregions in the boolean array. The ratio is therefore based on both the historic
    visits and the new set of candidate trajectories considered. Visiting a specific subregion of interest, 
    as defined by the tensor, will only give gain to the objetive once.

    Args:
        number_of_spacecrafts (int): Number of spacecraft
        spin_axis (np.ndarray): The axis around which the body rotates.
        spin_velocity (float): Angular velocity of the body's rotation.
        positions (np.ndarray): (3,N) Array of positions (in cartesian coordinates).
        velocities (np.ndarray): (3,N) Array of positions (in cartesian frame).
        timesteps (np.ndarray): (N) Array of time values for each position.
        radius_min (float): Inner radius of spherical grid, typically radius_inner_bounding_sphere.
        radius_max (float): Outer radius of spherical grid, typically radius_outer_bounding_sphere.
        r (np.ndarray): Array of r coordinates for each point defined on the spherical tensor.
        theta (np.ndarray): Array of theta coordinates for each point defined on the spherical tensor.
        phi (np.ndarray): Array of phi coordinates for each point defined on the spherical tensor.
        bool_tensor (np.ndarray): Boolean tensor corresponding to each point defined on the spherical grid.

    Returns:
        fitness (float): Aggregate coverage of the spherical tensor grid for a set of active trajectories (where coverage = ratio of visited points + weights). 
    """
    # Rotate positions according to body's rotation to simulate that the grid (i.e gravitational field approximation) is also rotating accordingly.
    rotated_positions = None

    # Check if only given a single position (Special case for tests).
    if positions.ndim == 1:
        rotated_positions = rotate_point(timesteps[0], positions, spin_axis, spin_velocity)
        r_points, theta_points, phi_points = cart2sphere(rotated_positions[0], rotated_positions[1], rotated_positions[2])

        if (r_points > radius_max) or (r_points < radius_min):
            r_points = []
        else: 
            r_points = [r_points]
            theta_points = [theta_points]
            phi_points = [phi_points]

    else:
        pos = np.array_split(positions, number_of_spacecrafts, axis=1)
        for counter, pos_arr in enumerate(pos):

            rot_pos_arr = np.empty((pos_arr.shape))

            for col in range(0,len(pos_arr[0,:])):
                rot_pos_arr[:,col] = rotate_point(timesteps[col], pos_arr[:,col], spin_axis, spin_velocity)

            if counter == 0:
                rotated_positions = rot_pos_arr
            else:
                rotated_positions = np.hstack((rotated_positions, rot_pos_arr))

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
        # NOTE: (Special case) No valid positions along candidate trajectory. 
        # Return fitness of previous visits instead (no new gain).
        indices_previous_trajectory  =  np.where(bool_tensor == True)
        n_previous_visits = indices_previous_trajectory[0].shape[0]
        ratio_previous_visits = n_previous_visits / bool_tensor.size
        sum_of_weights = np.sum(1/r[indices_previous_trajectory[0]])
        fitness = ratio_previous_visits + sum_of_weights
        return fitness
    
    else: 
        # Find the indices of the closest values in the meshgrid for each point using broadcasting.
        i = np.argmin(np.abs(r[:, np.newaxis] - r_points), axis=0) # indices along r axis
        j = np.argmin(np.abs(theta[:, np.newaxis] - theta_points), axis=0) # indices along theta axis
        k = np.argmin(np.abs(phi[:, np.newaxis] - phi_points), axis=0) # indices along phi axis

        # Create a new boolean tensor corresponding to the previous visits.
        new_tensor = np.full((len(r), len(theta), len(phi)), False)
        indices_previous_trajectory  =  np.where(bool_tensor == True)
        new_tensor[indices_previous_trajectory[0], indices_previous_trajectory[1], indices_previous_trajectory[2]] = True 
        
        # Update the new tensor with the proposed candidate trajectory.
        new_tensor[i, j, k] = True

        # Identify number of unique visits in the region of interest
        indices_unique_visits = np.where(new_tensor == True)
        n_unique_visits = indices_unique_visits[0].shape[0]

        # Compute ratio of uniquely visited regions to the number of regions defined by the tensor.
        ratio_unique_visits = n_unique_visits / bool_tensor.size

        # Get weights corresponding to the uniquely visited regions. 
        r_idx = indices_unique_visits[0] #np.concatenate((previous_visits[0], new_visits[0])) # unique_visits[0]
        sum_of_weights = np.sum(1/r[r_idx])

        # Return fitness
        fitness = ratio_unique_visits + sum_of_weights
        return fitness


def update_spherical_tensor_grid(number_of_spacecrafts: int, spin_axis: np.ndarray, spin_velocity: float, positions: np.ndarray, velocities: np.ndarray, timesteps: np.ndarray, radius_min: float, radius_max: float, r: np.ndarray, theta: np.ndarray, phi: np.ndarray, bool_tensor: np.ndarray) -> np.ndarray:
    """
    The function adjusts the trajectory for the body's sidereal rotation, identifies the 
    positions within the region of interest, matches these positions to the closest points on the
    spherical meshgrid and update the boolean tensor accordingly.

    Args:
        number_of_spacecrafts (int): Number of spacecraft
        spin_axis (np.ndarray): The axis around which the body rotates.
        spin_velocity (float): Angular velocity of the body's rotation.
        positions (np.ndarray): (3,N) Array of positions (in cartesian coordinates).
        velocities (np.ndarray): (3,N) Array of positions (in cartesian frame).
        timesteps (np.ndarray): (N) Array of time values for each position.
        radius_min (float): Inner radius of spherical grid, typically radius_inner_bounding_sphere.
        radius_max (float): Outer radius of spherical grid, typically radius_outer_bounding_sphere.
        r (np.ndarray): Array of r coordinates for each point defined on the spherical tensor.
        theta (np.ndarray): Array of theta coordinates for each point defined on the spherical tensor.
        phi (np.ndarray): Array of phi coordinates for each point defined on the spherical tensor.
        bool_tensor (np.ndarray): Boolean tensor corresponding to each point defined on the spherical grid.

    Returns:
        bool_tensor (np.ndarray): Updated boolean tensor corresponding to each point defined on the spherical grid.
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

    # Update and return boolean tensory given information on the new trajectory
    if len(r_points)>0 and len(theta_points)>0 and len(phi_points)>0:
        # Find the indices of the closest values in the meshgrid for each point using broadcasting
        i = np.argmin(np.abs(r[:, np.newaxis] - r_points), axis=0) # indices along r axis
        j = np.argmin(np.abs(theta[:, np.newaxis] - theta_points), axis=0) # indices along theta axis
        k = np.argmin(np.abs(phi[:, np.newaxis] - phi_points), axis=0) # indices along phi axis

        # Update tensor with information on new trajectory using advanced indexing
        bool_tensor[i, j, k] = True
    return bool_tensor


def create_spherical_tensor_grid(time_step: int, radius_min: float, radius_max: float, max_velocity_scaling_factor: float, fixed_velocity: np.ndarray) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a number of points in each spherical axis, which together defines
    a spherical tensor grid that satisfies the Courant–Friedrichs–Lewy condition.
    The function also generates a corresponding boolean tensor initialized by False-values
    indicating none of the defined points have been visited by a spacecraft.

    Args:
        time_step (int): Fixed measurement period (i.e time step along trajectory).
        radius_min (float): Inner radius of spherical grid, typically radius_inner_bounding_sphere.
        radius_max (float): Outer radius of spherical grid, typically radius_outer_bounding_sphere.
        max_velocity_scaling_factor (float): Scales the magnitude of the fixed-valued maximal velocity and therefore also the grid spacing.
        fixed_velocity (np.ndarray): A sample velocity vector for a spacecraft on a feasible trajectory around the considered body.
    
    Returns:
        r (np.ndarray): Array of r coordinates for each point defined on the spherical tensor.
        theta (np.ndarray): Array of theta coordinates for each point defined on the spherical tensor.
        phi (np.ndarray): Array of phi coordinates for each point defined on the spherical tensor.
        bool_tensor (np.ndarray): Boolean array corresponding to each point defined on the spherical tensor.
    """
    assert (time_step > 0)
    assert (radius_min > 0)
    assert (radius_max > radius_min)
    assert (max_velocity_scaling_factor > 0)

    # Fixed maximal velocity from previously defined trajectory. 
    # We use the fixed_velocity to avoid prioritizing higher velocities. 
    #
    # NOTE: The fitness value for covered volume will not be feasible if
    #        its maximal velocity exceeds the provided fixed_velocity as the grid
    #        spacing will be too small. Please use the scaling factor to adapt for this.

    # Define frequency of points for the spherical meshgrid: (see: Courant–Friedrichs–Lewy condition)
    max_velocity = np.max(np.linalg.norm(fixed_velocity)) * max_velocity_scaling_factor
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
