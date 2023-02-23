# General
import numpy as np
from typing import Union

# For working with the mesh
import mesh_utility

# For computing trajectory
from Integrator import Integrator

# For computing the next state
from Equations_of_motion import Equations_of_motion

# For Plotting
import pyvista as pv

# D-solver
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

    def __init__(self, body_density, target_altitude, final_time, start_time, time_step, lower_bounds, upper_bounds, algorithm):
        """ Setup udp attributes.

        Args:
            body_density (float):    Mass density of body of interest
            target_altitude (float): Target altitude for satellite trajectory. 
            final_time (int):        Final time for integration.
            start_time (int):        Start time for integration of trajectory (often zero)
            time_step (int):         Step size for integration. 
            lower_bounds (float):    Lower bounds for domain of initial state.
            upper_bounds (float):    Upper bounds for domain of initial state. 
            algorithm (str):         User defined algorithm of choice
        """
        # Creating the mesh (TetGen)
        self.body_mesh, self.mesh_vertices, self.mesh_faces = mesh_utility.create_mesh()

        # Assertions:
        assert body_density > 0
        assert target_altitude > 0
        assert final_time > start_time
        assert time_step <= (final_time - start_time)
        assert lower_bounds.all() < upper_bounds.all()

        # Setup equations of motion class
        self.eq_of_motion = Equations_of_motion(self.mesh_vertices, self.mesh_faces, body_density)

        # Setup user defined numerical integrator
        self.integrator = Integrator(final_time, start_time, time_step, algorithm, self.eq_of_motion)

        # Additional hyperparameters
        self.target_altitude = target_altitude     
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    
        #*********** HEYOKA TEST ************#
        self.body_density = body_density
        self.start_time = start_time
        self.final_time = final_time
        self.time_step = time_step
        #************************************#

    def fitness(self, x: np.ndarray) -> float:
        """ fitness evaluates the proximity of the satallite to target altitude.

        Args:
            x: State vector containing values for position and velocity of satelite in three dimensions. 

        Returns:
            fitness value (float): Difference between squared values of current and target altitude of satellite.
        """
        fitness_value, _ = self.compute_trajectory(np.array(x))
        return [fitness_value]


    def get_bounds(self) -> Union[np.ndarray, np.ndarray]:
        """get_bounds returns upper and lower bounds for the domain of the state vector.

        Returns:
            Two one-dimensional arrays for the bounady values of the state vector. 
        """
        return (self.lower_bounds, self.upper_bounds)

    def compute_trajectory(self, x: np.ndarray) -> Union[float, np.ndarray]:
        """compute_trajectory computes trajectory of satellite using numerical integation techniques 

        Args:
            x: State vector (position and velocity)

        Returns:
            fintess_values: Evaluation of proximity of satelite to target altitude.
            trajectory_info: Numpy array containing information on position and velocity at every time step.
        """

        # Fitness value (to be maximized)
        fitness_value = 0

        # Integrate trajectory
        initial_state = D.array(x)
        a = de.OdeSystem(
            self.eq_of_motion.compute_motion, 
            y0 = initial_state, 
            dense_output = True, 
            t = (self.start_time, self.final_time), 
            dt = self.time_step, 
            rtol = 1e-12, 
            atol = 1e-12)
        a.method = "RK87"
        a.integrate()
        trajectory_info = np.transpose(a.y)

        # Return fitness value for the computed trajectory
        squared_altitudes = trajectory_info[0,:]**2 + trajectory_info[1,:]**2 + trajectory_info[2,:]**2
        fitness_value = np.mean(np.abs(squared_altitudes-self.target_altitude))
        return fitness_value, trajectory_info 


    def plot_trajectory(self, r_store: np.ndarray):
        """plot_trajectory plots the body mesh and satellite trajectory.

        Args:
            r_store: Array containing values on position at each time step for the trajectory.
        """

        # Plotting mesh of asteroid/comet
        mesh_plot = pv.Plotter(window_size=[500, 500])
        mesh_plot.add_mesh(self.body_mesh.grid, show_edges=True)
        mesh_plot.show_grid() #grid='front',location='outer',all_edges=True 

        # Plotting trajectory
        trajectory_plot = np.transpose(r_store)
        mesh_plot.add_lines(trajectory_plot, color="red", width=40)
        #mesh_plot.add_mesh(trajectory_plot, style='wireframe', line_width=20, color='red')
        

        trajectory_plot = pv.PolyData(np.transpose(r_store[:,-1]))
        mesh_plot.add_mesh(trajectory_plot, color=[1.0, 1.0, 1.0], style='surface')
        
        mesh_plot.show(jupyter_backend = 'panel') 