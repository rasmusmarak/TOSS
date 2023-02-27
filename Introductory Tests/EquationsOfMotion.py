# General
import numpy as np

# For computing acceleration and potential
import polyhedral_gravity as model


class EquationsOfMotion:
    """
    Defining the celestial body of interest as a mesh object, as well as providing methods 
    for computing the satellite's acceleration and motion with respect to the celestial
    body and its current position expressed in three dimensions.
    """

    def __init__(self, mesh_vertices, mesh_faces, body_density):
        """
        Construtor for defining the celestial body as a mesh-based object.

        Args:
            mesh_vertices (_array_): Array containing all vertices of the mesh.
            mesh_faces (_array_): Array containing all the faces on the mesh.
            body_density (_float_): Density of the celestial body of interest represented by the mesh.
        """
        # Attributes relating to mesh 
        self.mesh_vertices = mesh_vertices
        self.mesh_faces = mesh_faces 
        self.body_density = body_density

    def compute_acceleration(self, x: np.ndarray) -> np.ndarray:
        """ 
        Computes acceleration at a given point with respect to a mesh representing
        a celestial body of interest. This is done using the Polyhedral-Gravity-Model package.

        Args:
            x (_np.ndarray_): Vector containing information on current position expressed in three dimensions (point).

        Returns:
            (_np.ndarray_): The acceleration at the given point x with respect to the mesh (celestial body).
        """
        _, a, _ = model.evaluate(self.mesh_vertices, self.mesh_faces, self.body_density, x)
        return -np.array(a)

    # Used by all RK-type algorithms
    def compute_motion(self, t: float, x: np.ndarray, risk_zone_radius: float = None) -> np.ndarray:
        """ State update equation for RK-type algorithms. 

        Args:
            t (_float_): Time value corresponding to current state
            x (_np.ndarray_): State vector containing position and velocity expressed in three dimensions.
            risk_zone_radius (_float_): Radius of bounding sphere around mesh. 

        Returns:
            (_np.ndarray_): K vector used for computing state at the following time step.
        """
        a = self.compute_acceleration(x[0:3])
        kx = x[3:6]  
        kv = a 
        return np.concatenate((kx, kv))

