# General
import numpy as np
from typing import Union

# For Plotting
import pyvista as pv

# For working with the mesh
import mesh_utility

# For computing the next state
from Equations_of_motion import Equations_of_motion

# For choosing numerical integration method
from Integrator import IntegrationScheme

# D-solver (performs integration)
import desolver as de
import desolver.backend as D
D.set_float_fmt('float64')



# Class representing UDP
class udp_initial_condition:
    """ 
    Sets up the user defined problem (udp) for use with pygmo.
    The object holds attributes in terms of variables and constants that
    are used for trajectory propagation. 
    The methods of the class defines the objective function for the optimization problem,
    boundaries for input variables, trajectory propagation and plotting of results. 
    """

    def __init__(self, body_density, target_altitude, final_time, start_time, time_step, lower_bounds, upper_bounds, algorithm, radius_bounding_sphere):
        """ Setup udp attributes.

        Args:
            body_density (_float_): Mass density of body of interest
            target_altitude (_float_): Target altitude for satellite trajectory. 
            final_time (_float_): Final time for integration.
            start_time (_float_): Start time for integration of trajectory (often zero)
            time_step (_float_): Step size for integration. 
            lower_bounds (_np.ndarray_): Lower bounds for domain of initial state.
            upper_bounds (_np.ndarray_): Upper bounds for domain of initial state. 
            algorithm (_int_): User defined algorithm of choice
            radius_bounding_sphere (_float_)_: Radius for the bounding sphere around mesh.
        """
        # Creating the mesh (TetGen)
        self.body_mesh, self.mesh_vertices, self.mesh_faces, largest_body_protuberant = mesh_utility.create_mesh()

        # Assertions:
        assert body_density > 0
        assert target_altitude > 0
        assert final_time > start_time
        assert time_step <= (final_time - start_time)
        assert lower_bounds.all() < upper_bounds.all()
        assert radius_bounding_sphere > largest_body_protuberant

        # Setup equations of motion class
        self.eq_of_motion = Equations_of_motion(self.mesh_vertices, self.mesh_faces, body_density)

        # Additional hyperparameters
        self.start_time = start_time
        self.final_time = final_time
        self.time_step = time_step
        self.target_altitude = target_altitude     
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.algorithm = algorithm
        self.radius_bounding_sphere = radius_bounding_sphere

    def fitness(self, x: np.ndarray) -> float:
        """ fitness evaluates the proximity of the satallite to target altitude.

        Args:
            x (_np.ndarray_): State vector containing values for position and velocity of satelite in three dimensions. 

        Returns:
            fitness value (_float_): Difference between squared values of current and target altitude of satellite.
        """
        fitness_value, _ = self.compute_trajectory(np.array(x))
        return [fitness_value]


    def get_bounds(self) -> Union[np.ndarray, np.ndarray]:
        """get_bounds returns upper and lower bounds for the domain of the state vector.

        Returns:
            lower_bounds (_np.ndarray_): Lower boundary values for the initial state vector.
            upper_bounds (_np.ndarray_): Lower boundary values for the initial state vector.
        """
        return (self.lower_bounds, self.upper_bounds)

    def compute_trajectory(self, x: np.ndarray) -> Union[float, np.ndarray]:
        """compute_trajectory computes trajectory of satellite using numerical integation techniques 

        Args:
            x (_np.ndarray_): State vector containing values for position and velocity of satelite in three dimensions.

        Returns:
            fitness_value (_float_): Evaluation of proximity of satelite to target altitude.
            trajectory_info (_np.ndarray_): Numpy array containing information on position and velocity at every time step (columnwise).
        """

        # Fitness value (to be maximized)
        fitness_value = 0

        print("Current x: ", x)

        # Integrate trajectory
        initial_state = D.array(x)
        a = de.OdeSystem(
            self.eq_of_motion.compute_motion, 
            y0 = initial_state, 
            dense_output = True, 
            t = (self.start_time, self.final_time), 
            dt = self.time_step, 
            rtol = 1e-12, 
            atol = 1e-12,
            constants=dict(risk_zone_radius = self.radius_bounding_sphere, mesh_vertices = self.mesh_vertices, mesh_faces = self.mesh_faces))
        a.method = str(IntegrationScheme(self.algorithm).name)

        check_for_collision.is_terminal = True
        a.integrate(events=check_for_collision)
        #a.integrate()
        trajectory_info = np.vstack((np.transpose(a.y), a.t))

        # Compute average distance to target altitude
        squared_altitudes = trajectory_info[0,:]**2 + trajectory_info[1,:]**2 + trajectory_info[2,:]**2
    
        # Add collision penalty
        if trajectory_info[-1,-1] < self.final_time:
            collision_penalty = 1e30
        else:
            collision_penalty = 0
        
        print("Reached time: ", trajectory_info[-1,-1])

        # Return fitness value for the computed trajectory
        fitness_value = np.mean(np.abs(squared_altitudes-self.target_altitude)) + collision_penalty
        return fitness_value, trajectory_info 



    def plot_trajectory(self, r_store: np.ndarray):
        """plot_trajectory plots the body mesh and satellite trajectory.

        Args:
            r_store (_np.ndarray_): Array containing values on position at each time step for the trajectory (columnwise).
        """

        # Plotting mesh of asteroid/comet
        mesh_plot = pv.Plotter(window_size=[500, 500])
        mesh_plot.add_mesh(self.body_mesh.grid, show_edges=True)
        mesh_plot.show_grid() #grid='front',location='outer',all_edges=True 

        # Plotting trajectory
        trajectory_plot = np.transpose(r_store)
        if (len(trajectory_plot[:,0]) % 2) != 0:
            trajectory_plot = trajectory_plot[0:-1,:,]
        mesh_plot.add_lines(trajectory_plot[:,0:3], color="red", width=40)        

        # Plotting final position as a white dot
        trajectory_plot = pv.PolyData(np.transpose(r_store[-1,0:3]))
        mesh_plot.add_mesh(trajectory_plot, color=[1.0, 1.0, 1.0], style='surface')
        
        mesh_plot.show(jupyter_backend = 'panel') 



def check_for_collision(t: float, state: np.ndarray, risk_zone_radius: float, mesh_vertices: np.ndarray, mesh_faces: np.ndarray) -> float:
    """ Checks for event: collision with the celestial body.

    Args:
        t (_float_): Current time step for integration.
        state (_np.ndarray_): Current state, i.e position and velocity
        risk_zone_radius (_float_): Radius of bounding sphere around mesh. 

    Returns:
        (_float_): Either distance from origin to satellite for collisions, or zero for no collisions.
    """
    position = state[0:3]
    distance = risk_zone_radius - D.norm(position)
    
    # If satellite is within risk-zone
    if distance >= 0:
        collision_avoided = point_is_outside_mesh(position, mesh_vertices, mesh_faces)

        # If there is a collision with the celestial body, return 0:
        if collision_avoided.all() == False:
            return 0
    
    # Else, return 1
    return 1


def point_is_outside_mesh(x: np.ndarray, mesh_vertices: np.ndarray, mesh_faces: np.ndarray) -> bool:
    """
    Uses is_outside to check if a set of positions (or current) x is is inside mesh.
    Returns boolean with corresponding results.

    Args:
        x (_np.ndarray_): Array containing current, or a set of, positions expressed in 3 dimensions.

    Returns:
        collision_boolean (_bool): A one dimensional array with boolean values corresponding to each
                                position kept in x. Returns "False" if point is inside mesh, and 
                                "True" if point is outside mesh (that is, there no collision).
    """
    collision_boolean = mesh_utility.is_outside(x, mesh_vertices, mesh_faces)
    return collision_boolean

