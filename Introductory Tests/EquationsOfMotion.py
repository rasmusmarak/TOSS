# General
import numpy as np
from math import pi, radians

# For computing acceleration and potential
import polyhedral_gravity as model

# For computing rotations of orbits
from pyquaternion import Quaternion


class EquationsOfMotion:
    """
    Defining the celestial body of interest as a mesh object, as well as providing methods 
    for computing the satellite's acceleration and motion with respect to the celestial
    body and its current position expressed in three dimensions.
    """

    def __init__(self, mesh_vertices, mesh_faces, body_args):
        """
        Construtor for defining the celestial body as a mesh-based object.

        Args:
            mesh_vertices (array): Array containing all vertices of the mesh.
            mesh_faces (array): Array containing all the faces on the mesh.
            body_args (np.ndarray): Paramteers relating to the celestial body:
                [0] body_density (float): Body density of celestial body.
                [1] body_mu (float): Gravitational parameter for celestial body.
                [2] body_declination (float): Declination angle of spin axis.
                [3] body_right_acension (float): Right ascension angle of spin axis.
                [4] body_spin_period (float): Rotational period around spin axis of the body.
        """
        # Assertion
        assert body_args[0] > 0
        assert body_args[2] >= 0
        assert body_args[3] >= 0
        assert body_args[4] >= 0

        # Attributes relating to mesh 
        self.mesh_vertices = mesh_vertices
        self.mesh_faces = mesh_faces 

        # Attributes relating to body 
        self.body_density = body_args[0]
        self.body_declination = body_args[2]
        self.body_right_acension = body_args[3]
        self.body_spin_period = body_args[4]

        # Setup spin axis of the body
        q_dec = Quaternion(axis=[1,0,0], angle=radians(self.body_declination)) # Rotate spin axis according to declination
        q_ra = Quaternion(axis=[0,0,1], angle=radians(self.body_right_acension)) # Rotate spin axis accordining to right ascension
        q_axis = q_dec * q_ra  # Composite rotation of q1 then q2 expressed as standard multiplication
        self.spin_axis = q_axis.rotate([0,0,1])

    def compute_acceleration(self, x: np.ndarray) -> np.ndarray:
        """ 
        Computes acceleration at a given point with respect to a mesh representing
        a celestial body of interest. This is done using the Polyhedral-Gravity-Model package.

        Args:
            x (np.ndarray): Vector containing information on current position expressed in three dimensions (point).

        Returns:
            (np.ndarray): The acceleration at the given point x with respect to the mesh (celestial body).
        """
        _, a, _ = model.evaluate(self.mesh_vertices, self.mesh_faces, self.body_density, x)
        return -np.array(a)

    # Used by all RK-type algorithms
    def compute_motion(self, t: float, x: np.ndarray, risk_zone_radius: float = None) -> np.ndarray:
        """ State update equation for RK-type algorithms. 

        Args:
            t (float): Time value corresponding to current state
            x (np.ndarray): State vector containing position and velocity expressed in three dimensions.
            risk_zone_radius (float): Radius of bounding sphere around mesh. 

        Returns:
            (np.ndarray): K vector used for computing state at the following time step.
        """
        rotated_position = self.orbit_rotation(t, x[0:3])
        a = self.compute_acceleration(rotated_position)
        kx = x[3:6]  
        kv = a 
        return np.concatenate((kx, kv))
    

    def orbit_rotation(self, t: float, x: np.ndarray) -> np.ndarray:
        """ Rotates position x according to the analyzed body's real rotation.
        Args:
            t (float): Time value when position x occurs.
            x (np.ndarray): Position of satellite expressed in three dimensions.
        Returns:
            x_rotated (np.ndarray): Rotated position using quaternions.
        """
        
        # Compute angular velocity of spin using known rotational period.
        spin_velocity = (2*pi)/self.body_spin_period

        # Get Quaternion object for rotation around spin axis
        q_rot = Quaternion(axis=self.spin_axis, angle=((2*pi)-(spin_velocity*t)))
        
        # Rotate satellite position using q_rot
        x_rotated = q_rot.rotate(x)

        return np.array(x_rotated)

