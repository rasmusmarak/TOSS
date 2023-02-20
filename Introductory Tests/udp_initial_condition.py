# General
import numpy as np

# For computing acceleration and potential
import polyhedral_gravity as model

# For working with the mesh
import mesh_utility

# For Plotting
import pyvista as pv


class udp_initial_condition:
    """ _summary_
    Sets up the user defined problem (udp) for use with pygmo.
    The object holds attributes in terms of variables and constants that
    are used for trajectory propagation. 
    The methods of the class defines the objective function for the optimization problem,
    boundaries for input variables, trajectoru propagation and plotting of results. 

    Dependencies:
        - Numpy
        - polyhedral-gravity-model
        - PyVista
    """

    def __init__(self, body_density, target_altitude, final_time, start_time, time_step, lower_bounds, upper_bounds):
        """__init__ _summary_
        Setup udp attributes.

        Args:
            body_density (float):    Mass density of body of interest
            target_altitude (float): Target altitude for satellite trajectory. 
            final_time (int):        Final time for integration.
            start_time (int):        Start time for integration of trajectory (often zero)
            time_step (int):         Step size for integration. 
            lower_bounds (float):    Lower bounds for domain of initial state.
            upper_bounds (float):    Upper bounds for domain of initial state. 
        """
        # Creating the mesh (TetGen)
        self.body_mesh, self.mesh_vertices, self.mesh_faces = mesh_utility.create_mesh()

        # Additional hyperparameters
        self.body_density = body_density     
        self.target_altitude = target_altitude     
        self.final_time = final_time      
        self.start_time = start_time                
        self.time_step = time_step
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds           
    
    def fitness(self,x):
        """ fitness evaluates the proximity of the satallite to target altitude.

        Args:
            x: State vector containing values for position and velocity of satelite in three dimensions. 

        Returns:
            fitness value (float): Difference between squared values of current and target altitude of satellite.
        """
        fitness_value, _, _, _ = self.compute_trajectory(x)
        return [fitness_value]

    def get_bounds(self):
        """get_bounds returns upper and lower bounds for the domain of the state vector.

        Returns:
            Two one-dimensional arrays for the bounady values of the state vector. 
        """
        return (self.lower_bounds, self.upper_bounds)

    def compute_trajectory(self, x):
        """compute_trajectory computes trajectory of satellite using numerical integation techniques 

        Args:
            x: State vector (position and velocity)

        Returns:
            fintess_values: Evaluation of proximity of satelite to target altitude.
            r_store:        Array containing values on position at each time step for the trajectory.
            v_store:        Array containing values on velocities at each time step for the trajectory.
            a_store:        Array containing values on acceleration at each time step for the trajectory.
        """
        # Initial information
        r = np.transpose(x[0:3]) # Start Position
        v = np.transpose(x[3:6]) # Initial velocity

        # Array containing times for summation
        time_list = np.arange(self.start_time, self.final_time, self.time_step)

        # Numpy Arrays to store trajectory information
        r_store = np.zeros((3,len(time_list)))
        v_store = np.zeros((3,len(time_list)))
        a_store = np.zeros((3,len(time_list)))

        # Add starting position to memory
        r_store[:,0] = r
        v_store[:,0] = v

        # Fitness value (to be maximized)
        fitness_value = 0
        collision = 0

        if r[0]**2 + r[1]**2 + r[2]**2 <= 5000**2:
            fitness_value = 1e30
            r_store = 0
            v_store = 0
            a_store = 0

        else:
            # Numerical integration of Newton's equations of motion (trajectory propagation)
            r_store, v_store, a_store, collision = self.euler_approx(r, v, time_list, r_store, v_store, a_store)

            if collision == 1:
                fitness_value = 1e30

            else:
                # Return fitness value for the computed trajectory
                squared_altitudes = r_store[0,:]**2 + r_store[1,:]**2 + r_store[2,:]**2
                fitness_value = np.mean(np.abs(squared_altitudes-self.target_altitude))
                
        return fitness_value, r_store, v_store, a_store

    
    def euler_approx(self, r, v, time_list, r_store, v_store, a_store):
        """euler_approx uses euler's method as numerical integrator for approximating the trajectory. 

        Args:
            r: Current position, expressed in thee dimensions.
            time_list: List of all the times corresponding to a state.
            r_store:   Array containing values on position at each time step for the trajectory.
            v_store:   Array containing values on velocity at each time step for the trajectory.
            a_store:   Array containing values on acceleration at each time step for the trajectory.

        Returns:
            Complete trajectory information stored in r_store, v_store and a_store.
        """
        for i in range(1,len(time_list)):
            # Retrieve information at current position
            _, a, _ = model.evaluate(self.mesh_vertices, self.mesh_faces, self.body_density, r)
            a = - np.array(a)

            # Computing velocity and position for next time-step
            v_n = v + self.time_step * a
            r_n = r + self.time_step * v_n

            # Update current velocity and position
            v = v_n
            r = r_n

            # Storing updated trajectory information
            r_store[:,i] = r
            v_store[:,i] = v
            a_store[:,i-1] = a
            i += 1

            collision = 0
            
            if r[0]**2 + r[1]**2 + r[2]**2 <= 5000*2:
                r_store = 0
                v_store = 0
                a_store = 0
                collision = 1
                return r_store, v_store, a_store, collision

        return r_store, v_store, a_store, collision


    def point_is_outside_mesh(self, x):
        """
        Uses is_outside to check if a set of positions (or current) x is is inside mesh.
        Returns boolean with corresponding results.

        Args:
            x: Array containing current, or a set of, positions expressed in 3 dimensions.

        Returns:
            collision_boolean: A one dimensional array with boolean values corresponding to each
                               position kept in x. Returns "False" if point is inside mesh, and 
                               "True" if point is outside mesh (that is, there no collision).
        """
        collision_boolean = mesh_utility.is_outside(x, self.mesh_vertices, self.mesh_faces)
        return collision_boolean


    def plot_trajectory(self, r_store):
        """plot_trajectory plots the body mesh and satellite trajectory.

        Args:
            r_store: Array containing values on position at each time step for the trajectory.
        """

        # Plotting mesh of asteroid/comet
        mesh_plot = pv.Plotter()
        mesh_plot.add_mesh(self.body_mesh.grid, show_edges=True)
        mesh_plot.show_grid(color='black') # Alternativ: show_bounds()

        # Plotting trajectory
        trajectory_plot = np.transpose(r_store)
        mesh_plot.add_lines(trajectory_plot, color="red", width=20)

        trajectory_plot = pv.PolyData(np.transpose(r_store[:,-1]))
        mesh_plot.add_mesh(trajectory_plot, color=[1.0, 1.0, 1.0], style='surface')
        
        #mesh_plot.set_background('grey') #Changes backrgound color of plot
        mesh_plot.show(jupyter_backend = 'panel' ) 