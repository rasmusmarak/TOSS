# core stuff
import numpy as np

# For Plotting
import matplotlib.pyplot as plt
import pyvista as pv


def fitness_over_generations(fitness_list, number_of_generations):
    """ Plots the champion fitness vlue of each generation

    Args:
        fitness_list (np.ndarray): Array containing champion fitness vlue of each generation.
        number_of_generations (int): Number of generations for the optimization.
    """
    # Plot fitness over generations
    figure, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(0, number_of_generations), fitness_list, label='Function value')
    champion_n = np.argmin(np.array(fitness_list))
    ax.scatter(champion_n, np.min(fitness_list), marker='x', color='r', label='All-time champion')
    ax.set_xlim((0, number_of_generations))
    ax.grid('major')
    ax.set_title('Best individual of each generation', fontweight='bold')
    ax.set_xlabel('Number of generation')
    ax.set_ylabel(r'UDP fitness value')
    ax.legend(loc='upper right')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.show()


def two_axis_trajectory(trajectory_info, axis_1, axis_2):
    """ Plots trajectory provided two axes of the trajectory.

    Args:
        trajectory_info (np.ndarray): Array containing values on position at each time step for the trajectory (columnwise).
        axis_1 (int): User provided axis
        axis_2 (int): User provided axis
    """
    # Plot a two axis trajectory
    figure, ax = plt.subplots()
    ax.plot(trajectory_info[axis_1,:],trajectory_info[axis_2,:])
    plt.show()


def plot_trajectory(r_store: np.ndarray, mesh):
    """plot_trajectory plots the body mesh and satellite trajectory.

    Args:
        r_store (np.ndarray): Array containing values on position at each time step for the trajectory (columnwise).
        mesh (tetgen.pytetgen.TetGen): Tetgen mesh object of celestial body.
    """
    # Plotting mesh of asteroid/comet
    mesh_plot = pv.Plotter(window_size=[500, 500])
    mesh_plot.add_mesh(mesh.grid, show_edges=True)
    #mesh_plot.show_bounds() # minor_ticks=True, grid='front',location='outer',all_edges=True 

    # Plotting trajectory
    trajectory = np.transpose(r_store)
    for i in range(0,len(r_store[0])-1):
        traj = np.vstack((trajectory[i,:], trajectory[i+1,:]))
        mesh_plot.add_lines(traj, color="red", width=40)
                    
    # Plotting final position as a white dot
    trajectory_plot = pv.PolyData(np.transpose(r_store[:,-1]))
    mesh_plot.add_mesh(trajectory_plot, color=[1.0, 1.0, 1.0], style='surface')

    mesh_plot.add_axes(x_color='red', y_color='green', z_color='blue', xlabel='X', ylabel='Y', zlabel='Z', line_width=2, shaft_length = 10)
    
    mesh_plot.show(jupyter_backend = 'panel') 