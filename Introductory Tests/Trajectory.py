# General
import numpy as np
from typing import Union

# For Plotting
import pyvista as pv

# For working with the mesh
import mesh_utility

# For computing the next state
from EquationsOfMotion import EquationsOfMotion

# For choosing numerical integration method
from Integrator import IntegrationScheme

# D-solver (performs integration)
import desolver as de
import desolver.backend as D
D.set_float_fmt('float64')


class Trajectory:
    """
    Holds all atributes, methods and functions related to integrating a trajectory
    given an intitial state, mesh of a celestial body and the integration library
    Desolver. The class also calls for files related to plotting the trajectory.
    """

    def __init__(self, body_density, final_time, start_time, time_step, algorithm, radius_bounding_sphere):
        """ Setup udp attributes.

        Args:
            body_density (float): Mass density of body of interest
            final_time (float): Final time for integration.
            start_time (float): Start time for integration of trajectory (often zero)
            time_step (float): Step size for integration. 
            algorithm (int): User defined algorithm of choice
            radius_bounding_sphere (float)_: Radius for the bounding sphere around mesh.
        """
        # Creating the mesh (TetGen)
        self.body_mesh, self.mesh_vertices, self.mesh_faces, largest_body_protuberant = mesh_utility.create_mesh()

        # Assertions:
        assert body_density > 0
        assert final_time > start_time
        assert time_step <= (final_time - start_time)
        assert radius_bounding_sphere > largest_body_protuberant

        # Setup equations of motion class
        self.eq_of_motion = EquationsOfMotion(self.mesh_vertices, self.mesh_faces, body_density)

        # Additional hyperparameters
        self.start_time = start_time
        self.final_time = final_time
        self.time_step = time_step
        self.algorithm = algorithm
        self.radius_bounding_sphere = radius_bounding_sphere


    def integrate(self, x: np.ndarray) -> Union[np.ndarray, float, float]:
        """compute_trajectory computes trajectory of satellite using numerical integation techniques 

        Args:
            x (np.ndarray): State vector containing values for position and velocity of satelite in three dimensions.

        Returns:
            trajectory_info (np.ndarray): Numpy array containing information on position and velocity at every time step (columnwise).
            squared_altitudes (float): Sum of squared altitudes above origin for every position
            collision_penalty (float): Penalty value given for the event of a collision with the celestial body.
        """

        # Integrate trajectory
        initial_state = D.array(x)
        trajectory = de.OdeSystem(
            self.eq_of_motion.compute_motion, 
            y0 = initial_state, 
            dense_output = True, 
            t = (self.start_time, self.final_time), 
            dt = self.time_step, 
            rtol = 1e-12, 
            atol = 1e-12,
            constants=dict(risk_zone_radius = self.radius_bounding_sphere)) #, mesh_vertices = self.mesh_vertices, mesh_faces = self.mesh_faces
        trajectory.method = str(IntegrationScheme(self.algorithm).name)

        point_is_inside_risk_zone.is_terminal = False
        trajectory.integrate(events=point_is_inside_risk_zone)
        trajectory_info = np.vstack((np.transpose(trajectory.y), trajectory.t))

        # Compute average distance to target altitude
        squared_altitudes = trajectory_info[0,:]**2 + trajectory_info[1,:]**2 + trajectory_info[2,:]**2

        # Add collision penalty
        points_inside_risk_zone = np.empty((len(trajectory.events), 3), dtype=np.float64)
        i = 0
        for j in trajectory.events:
            points_inside_risk_zone[i,:] = j.y[0:3]
            i += 1
        
        collision_avoided = point_is_outside_mesh(points_inside_risk_zone, self.mesh_vertices, self.mesh_faces)
        if all(collision_avoided) == True:
            collision_penalty = 0
        else:
            collision_penalty = 1e30
        
        # Return trajectory and neccessary values for computing fitness in udp.
        return trajectory_info, squared_altitudes, collision_penalty



    def plot_trajectory(self, r_store: np.ndarray):
        """plot_trajectory plots the body mesh and satellite trajectory.

        Args:
            r_store (np.ndarray): Array containing values on position at each time step for the trajectory (columnwise).
        """
        # Plotting mesh of asteroid/comet
        mesh_plot = pv.Plotter(window_size=[500, 500])
        mesh_plot.add_mesh(self.body_mesh.grid, show_edges=True)
        mesh_plot.show_bounds(minor_ticks=True) #grid='front',location='outer',all_edges=True 

        # Plotting trajectory
        trajectory = np.transpose(r_store)
        for i in range(0,len(r_store[0])-1):
            traj = np.vstack((trajectory[i,:], trajectory[i+1,:]))
            mesh_plot.add_lines(traj, color="red", width=40)
                        
        # Plotting final position as a white dot
        trajectory_plot = pv.PolyData(np.transpose(r_store[:,-1]))
        mesh_plot.add_mesh(trajectory_plot, color=[1.0, 1.0, 1.0], style='surface')

        
        mesh_plot.show(jupyter_backend = 'panel') 



def point_is_inside_risk_zone(t: float, state: np.ndarray, risk_zone_radius: float) -> int:
    """ Checks for event: collision with the celestial body.

    Args:
        t (float): Current time step for integration.
        state (np.ndarray): Current state, i.e position and velocity
        risk_zone_radius (float): Radius of bounding sphere around mesh. 

    Returns:
        (int): Returns 1 when the satellite enters the risk-zone, and 0 otherwise.
    """
    position = state[0:3]
    distance = risk_zone_radius**2 - position[0]**2 + position[1]**2 + position[2]**2
    if distance >= 0:
        return 0
    return 1


def point_is_outside_mesh(x: np.ndarray, mesh_vertices: np.ndarray, mesh_faces: np.ndarray) -> bool:
    """
    Uses is_outside to check if a set of positions (or current) x is is inside mesh.
    Returns boolean with corresponding results.

    Args:
        x (np.ndarray): Array containing current, or a set of, positions expressed in 3 dimensions.

    Returns:
        collision_boolean (bool): A one dimensional array with boolean values corresponding to each
                                position kept in x. Returns "False" if point is inside mesh, and 
                                "True" if point is outside mesh (that is, there no collision).
    """
    collision_boolean = mesh_utility.is_outside(x, mesh_vertices, mesh_faces)
    return collision_boolean

