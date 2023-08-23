# General
import numpy as np
from math import pi, radians

# For computing acceleration and potential
import polyhedral_gravity as model

# For computing rotations of orbits
from pyquaternion import Quaternion


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
    position = x[0:3]

    if args.problem.activate_rotation:
        rotated_position = rotate_point(t, x[0:3], args.body.spin_axis, args.body.spin_velocity)
        position = rotated_position

    a = compute_acceleration(position, args)

    kx = x[3:6]
    kv = a

    return np.concatenate((kx, kv))


def rotate_point(t: float, x: np.ndarray, spin_axis: np.ndarray, spin_velocity: float) -> np.ndarray:
    """ Rotates position x according to the analyzed body's real rotation.
        The rotation is made in the 3D cartesian inertial body frame.
    Args:
        t (float): Time value in seconds when position x occurs.
        x (np.ndarray): Position of satellite expressed in the 3D cartesian coordinates.
        spin_axis (np.ndarray): The axis around which the body rotates.
        spin_velocity (float): Angular velocity of the body's rotation.
        
    Returns:
        x_rotated (np.ndarray): Rotated position of satellite expressed in the 3D cartesian coordinates.
    """

    # Get Quaternion object for rotation around spin axis
    q_rot = Quaternion(axis=spin_axis, angle=-(spin_velocity*t))
    
    # Rotate satellite position using q_rot
    x_rotated = q_rot.rotate(x)

    return np.array(x_rotated)

