# General
import numpy as np

# For computing acceleration and potential
import polyhedral_gravity as model


class Equations_of_motion:

    def __init__(self, mesh_vertices, mesh_faces, body_density):
        # Attributes relating to mesh 
        self.mesh_vertices = mesh_vertices
        self.mesh_faces = mesh_faces 
        self.body_density = body_density

    def compute_acceleration(self, x: np.ndarray) -> np.ndarray:
        _, a, _ = model.evaluate(self.mesh_vertices, self.mesh_faces, self.body_density, x)
        return -np.array(a)

    # Used by all RK-type algorithms
    def compute_motion(self, x: np.ndarray, t: float = None) -> np.ndarray:
        """ State update equation for RK-type algorithms. 

        Args:
            t : Time value corresponding to current state
            x : State vector containing position and velocity expressed in three dimensions.

        Returns:
            State vector used for computing state at the following time step.
        """
        a = self.compute_acceleration(x[0:3])
        kx = x[3:6]  
        kv = a 
        return np.concatenate((kx, kv))

