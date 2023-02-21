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

#********* HEYOKA *********#
# For computing acceleration and potential
import polyhedral_gravity as model
import heyoka as hk
#**************************#

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


    def comp_acc(self,point):
        _, a, _ = model.evaluate(self.mesh_vertices, self.mesh_faces, self.body_density, point)
        a = -np.array(a)
        return a

    def compute_trajectory(self, initial_state: np.ndarray) -> Union[float, np.ndarray]:
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
        #trajectory_info = self.integrator.run_integration(x)



        #************ HEYOKA TEST ************#

        # create heyoka variables
        x,y,z,vx,vy,vz = hk.make_vars("x","y","z","vx","vy","vz")


        #results = lambda x,y,z: model.evaluate(self.mesh_vertices, self.mesh_faces, self.body_density, [x,y,z])
        #a = -np.array(reults(1))

        _,a,_ = model.evaluate(self.mesh_vertices, self.mesh_faces, self.body_density, [x,y,z])
        a = -a

        # EOM
        dxdt = vx
        dydt = vy
        dzdt = vz
        
        dvxdt = a[0]
        dvydt = a[1]
        dvzdt = a[2]

        # Instantiate the heyoka (taylor) algorithm
        ta = hk.taylor_adaptive(sys = [(x,dxdt),(y,dydt),(z,dzdt),(vx,dvxdt),(vy,dvydt),(vz,dvzdt)],
                        state = initial_state,
                        time = self.start_time,
                        tol = 1e-16)


        # Here we redefine the initial conditions since we may want to change them without recompiling the integrator
        ta.time = self.start_time

        # Note the python syntax to assign directly the array elements. Here necessary
        # as heyoka does not allow to change the memory location of the state
        ta.state[:] = initial_state

        # Propagate trajectory over time interval tgrid
        tgrid = np.linspace(self.start_time, self.final_time ,int(1 + self.final_time/self.time_step), endpoint = True)
        trajectory_info = ta.propagate_grid(tgrid)

        trajectory_info = np.array(trajectory_info)
        trajectory_info = np.transpose(trajectory_info)

        #*************************************#



        
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
        mesh_plot = pv.Plotter()
        mesh_plot.add_mesh(self.body_mesh.grid, show_edges=True)
        mesh_plot.show_bounds(grid='front',location='outer',all_edges=True)

        # Plotting trajectory
        trajectory_plot = np.transpose(r_store)
        mesh_plot.add_lines(trajectory_plot, color="red", width=20)

        trajectory_plot = pv.PolyData(np.transpose(r_store[:,-1]))
        mesh_plot.add_mesh(trajectory_plot, color=[1.0, 1.0, 1.0], style='surface')
        
        mesh_plot.show(jupyter_backend = 'panel') 