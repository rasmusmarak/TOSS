# General
import numpy as np

# For computing acceleration and potential
import polyhedral_gravity as model

# For working with the mesh
import mesh_utility

# For Plotting
import pyvista as pv

class udp_obj:
    
    def __init__(self,density, r_T, t_end, t_0, dt, lb, ub):
        # Creating the mesh (TetGen)
        self.mesh, self.vertices, self.faces = mesh_utility.create_mesh()

        # Additional hyperparameters
        self.density = density     
        self.r_T = r_T     
        self.t_end = t_end      
        self.t_0 = t_0                
        self.dt = dt
        self.lb = lb
        self.ub = ub           
    
    def fitness(self,x):
        fit_val, _, _, _ = self.compute_trajectory(x)
        return [fit_val]

    def get_bounds(self):
        return (self.lb, self.ub)

    def compute_trajectory(self, x):
        # Initial information
        r = np.transpose(x[0:3]) # Start Position
        v = np.transpose(x[3:6]) # Initial velocity

        # Array containing times for summation
        time_list = np.arange(self.t_0, self.t_end, self.dt)

        # Numpy Arrays to store trajectory information
        r_store = np.zeros((3,len(time_list)))
        v_store = np.zeros((3,len(time_list)))
        a_store = np.zeros((3,len(time_list)))

        # Add starting position to memory
        r_store[:,0] = r
        v_store[:,0] = v

        # Fitness value (to be maximized)
        fit_val = 0

        for i in range(1,len(time_list)):
            # Retrieve information at current position
            _, a, _ = model.evaluate(self.vertices, self.faces, self.density, r)
            a = - np.array(a)

            # Computing velocity and position for next time-step
            v_n = v + self.dt * a
            r_n = r + self.dt * v_n

            # Update current velocity and position
            v = v_n
            r = r_n

            # Storing updated trajectory information
            r_store[:,i] = r
            v_store[:,i] = v
            a_store[:,i-1] = a
            i += 1

        altitudes = (r_store[0,:]**2 + r_store[1,:]**2 + r_store[2,:]**2)**(1/2)
        fit_val = np.mean(np.abs(altitudes-self.r_T))
        return fit_val, r_store, v_store, a_store


    def plot_trajectory(self, r_store):

        # Plotting mesh of asteroid/comet
        mesh_plot = pv.Plotter()
        mesh_plot.add_mesh(self.mesh.grid, show_edges=True)
        mesh_plot.show_bounds(grid='front',location='outer',all_edges=True)

        # Plotting trajectory
        trajectory_plot = np.transpose(r_store)
        mesh_plot.add_lines(trajectory_plot, color="red", width=20)

        trajectory_plot = pv.PolyData(np.transpose(r_store[:,-1]))
        mesh_plot.add_mesh(trajectory_plot, color=[1.0, 1.0, 1.0], style='surface')
        
        mesh_plot.show(jupyter_backend = 'panel')