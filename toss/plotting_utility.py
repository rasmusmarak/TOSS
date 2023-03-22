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
    ax.set_ylabel('UDP fitness value')
    ax.legend(loc='upper right')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig('figures/fitness_over_generations.png')


def two_axis_trajectory(trajectory_info, axis_1, axis_2):
    """ Plots trajectory provided two axes of the trajectory.

    Args:
        trajectory_info (np.ndarray): Array containing values on position at each time step for the trajectory (columnwise).
        axis_1 (int): User provided axis
        axis_2 (int): User provided axis
    """
    # Plot a two axis trajectory
    figure, (ax1, ax2, ax3) = plt.subplots(1,3)

    ax1.plot(trajectory_info[0,:],trajectory_info[1,:])
    ax1.set_title("Axis: (x,y)")

    ax2.plot(trajectory_info[0,:],trajectory_info[2,:])
    ax1.set_title("Axis: (x,z)")

    ax3.plot(trajectory_info[2,:],trajectory_info[3,:])
    ax1.set_title("Axis: (y,z)")

    plt.savefig('figures/two_axis_plot.png')

def plot_UDP(args, r_store: np.ndarray, plot_mesh, plot_trajectory, plot_risk_zone):
    """plot_trajectory plots the satellite trajectory.

    Args:
        (3xN) r_store (np.ndarray): Array containing N positions (cartesian frame) on the trajectory.
    """
    # Define figure
    #ax = plt.figure().add_subplot(projection='3d')
    fig = plt.figure(figsize = (13,7))
    ax = fig.add_subplot(projection='3d')

    #Plot trajectory
    if plot_trajectory:
        x = r_store[0,:]
        y = r_store[1,:]
        z = r_store[2,:]
        ax.plot(x, y, z, label='Trajectory')
        ax.legend()

    # Plot mesh:
    if plot_mesh:
        ax.plot_trisurf(args.mesh.vertices[:, 0], args.mesh.vertices[:,1], triangles=args.mesh.faces, Z=args.mesh.vertices[:,2], alpha=1, color='grey') 

    # Plot risk zone:
    if plot_risk_zone:
        r = args.problem.radius_bounding_sphere
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:40j]
        X = r*np.cos(u)*np.sin(v)
        Y = r*np.sin(u)*np.sin(v)
        Z = r*np.cos(v)
        ax.plot_wireframe(X, Y, Z, color="r", alpha=0.1)

    ax.set_title("Solution to UDP")
    ax.view_init(15,-45)

    # Make axes limits 
    xyzlim = np.array([ax.get_xlim3d(),ax.get_ylim3d(),ax.get_zlim3d()]).T
    XYZlim = [min(xyzlim[0]),max(xyzlim[1])]
    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim)

    plt.savefig('figures/trajectory_plot.png')


def plot_trajectory_with_mesh_pyvista(r_store: np.ndarray, mesh):
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
    
    #mesh_plot.show(jupyter_backend = 'panel')
    mesh_plot.save_graphic("figures/trajectory_mesh_plot.pdf")