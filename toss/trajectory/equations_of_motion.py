# General
import numpy as np
from math import pi, radians

# For computing acceleration and potential
import polyhedral_gravity as model

# For computing rotations of orbits
import quaternion


def setup_spin_axis(args):
    """ 
    Defines a spin axis given the declination and
    right ascension of the body's rotation.

    Args:
        args (dotmap.DotMap):
            body:
                declination (float): Declination angle of spin axis.
                right_ascension (float): Right ascension angle of spin axis.

    Returns:
        spin_axis (np.ndarray): The axis around which the body rotates.
    """
    # Convert spin axis orientation properties to rad
    rotation_declination = radians(args.body.declination)
    rotation_right_ascension = radians(args.body.right_ascension)
    
    # Define the rotation axis as a unit vector
    spin_axis = np.array(
        [
            np.cos(rotation_declination) * np.cos(rotation_right_ascension),
            np.cos(rotation_declination) * np.sin(rotation_right_ascension),
            np.sin(rotation_declination),
        ]
    )
    return spin_axis


def compute_acceleration(x: np.ndarray, args) -> np.ndarray:
    """ 
    Computes acceleration at a given point with respect to a mesh representing
    a celestial body of interest. This is done using the Polyhedral-Gravity-Model package.

    Args:
        x (np.ndarray): Vector containing information on current position expressed in 3D cartesian coordinates.
        args (dotmap.DotMap):
            body:
                density (float): Mass density of celestial body.
            mesh:
                vertices (np.ndarray): Array containing all points on mesh.
                faces (np.ndarray): Array containing all triangles on the mesh.

    Returns:
        (np.ndarray): The acceleration at the given point x with respect to the mesh (celestial body).
    """

    #x = np.asarray([x[0][0], x[1][0], x[2][0]], dtype=np.float64)
    #x_flat = np.ravel(x)
    #x_new = np.array([x_flat[0], x_flat[1], x_flat[2]], dtype=np.float64)
    #r_sq = np.dot(np.transpose(x_new),x_new)

    #if r_sq > 1e12:
        # Estimate gravitational attraction as point source to avoid numeric instability for polyhedral model at great distances (>1000 km),
    #    m = 1e13
    #    g = 6.67 * 10**(-11)
    #    theta = np.arctan(x_new[1]/x_new[0], dtype=np.float64)
    #    phi = np.arccos(x_new[2]/ r_sq**(1/2), dtype=np.float64)
    #    c = g*m/r_sq
    #    a = [c*np.cos(theta), c*np.sin(theta), c*np.cos(phi)]

    #else:
    _, a, _ = args.mesh.evaluable(x, args.integrator.parallel_acceleration_computation)
    return -np.array(a)

# Used by all RK-type algorithms
def compute_motion(t: float, x: np.ndarray, args) -> np.ndarray:
    """ State update equation for RK-type algorithms. 

    Args:
        t (float): Time value in seconds corresponding to current state
        x (np.ndarray): State vector containing position and velocity expressed in 3D cartesian coordinates.
        args (dotmap.DotMap):
            body:
                density (float): Mass density of celestial body.
                spin_velocity (float): Angular velocity of the body's rotation.
                spin_axis (np.ndarray): The axis around which the body rotates.
            problem:
                activate_rotation (bool): Activates/deactivates compution of motion with respect to the body's rotation.
            mesh:
                vertices (np.ndarray): Array containing all points on mesh.
                faces (np.ndarray): Array containing all triangles on the mesh.
            state: Parameters provided by the state vector
                time_of_maneuver (float): Time for adding impulsive maneuver [seconds].
                delta_v (np.ndarray): Array containing the cartesian componants of the impulsive maneuver.
    Returns:
        (np.ndarray): K vector used for computing state at the following time step.
    """

    # Aound variables in state vector to match the integrators tolerance level to avoid overflow causing numeric instability.
    x = np.around(x, decimals=int(-np.log10(args.integrator.atol)))

    # Adjust positions for chosen reference frame
    if args.problem.activate_rotation:
        rotated_position = rotate_point(t, x[0:3], args.body.spin_axis, args.body.spin_velocity, None)
        position = rotated_position
    else:
        position = x[0:3]

    # Compute accelertaion in chosen reference frame
    a = compute_acceleration(position, args)

    # Return Runge-Kutta parameters
    kx = x[3:6]
    kv = a

    return np.concatenate((kx, kv))


def rotate_point(t: float or list or np.ndarray, x: np.ndarray, spin_axis: np.ndarray, spin_velocity: float, quaternion_objects: np.ndarray or None) -> np.ndarray:
    """ Rotates position x according to the analyzed body's real rotation.
        The rotation is made in the body-fixed frame.
    Args:
        t (float or list or np.ndarray): Time value in seconds when position x occurs.
        x (np.ndarray): Position of satellite expressed in the 3D cartesian coordinates.
        spin_axis (np.ndarray): The axis around which the body rotates.
        spin_velocity (float): Angular velocity of the body's rotation.
        quaternion_objects (np.ndarray or None): Prepared quaternion objects for rotating complete trajetcory using predetermined angles.
        
    Returns:
        x_rotated (np.ndarray): Rotated position of satellite expressed in the 3D cartesian coordinates.
    """
    if quaternion_objects == None:
        if x.ndim == 1:  # Handle single position case
            x = x.reshape(3, 1)
            angles = [-spin_velocity*t]
        else:
            angles = -spin_velocity*np.asarray(t)

        # Create an array of quaternion objects for rotations
        rotations = np.array([quaternion.from_rotation_vector(angle * spin_axis) for angle in angles])
    else:
        rotations = quaternion_objects
 
    # Convert positions array to quaternion objects
    positions_quat = np.array([quaternion.quaternion(*pos) for pos in x.T])
    
    # Perform quaternion rotations element-wise
    rotated_positions_quat = rotations * positions_quat * rotations.conj()

    # Convert rotated quaternion positions back to arrays
    rotated_positions = np.array([quat.vec for quat in rotated_positions_quat]).T


    if rotated_positions.ndim == 1:  # Handle single position case
        return x.reshape(3)
    else:
        return rotated_positions


