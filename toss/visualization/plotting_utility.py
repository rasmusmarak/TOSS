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
    #plt.show()
    plt.savefig('figures/67P/fitness_over_generations.png')


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
    #plt.show()
    plt.savefig('figures/67P/two_axis_plot.png')



def plot_trajectory(positions: np.ndarray, mesh):
    """plot_trajectory plots the body mesh and satellite trajectory.

    Args:
        positions (np.ndarray): (3,N) Array containing position along computed trajectory (expressed in cartesian frame).
        mesh (tetgen.pytetgen.TetGen): Tetgen mesh object of celestial body.
    """
    # Plotting mesh of asteroid/comet
    mesh_plot = pv.Plotter(window_size=[500, 500])
    mesh_plot.add_mesh(mesh.grid, show_edges=True)
    #mesh_plot.show_bounds() # minor_ticks=True, grid='front',location='outer',all_edges=True 

    # Plotting trajectory
    trajectory = np.transpose(positions)
    for i in range(0,len(positions[0])-1):
        traj = np.vstack((trajectory[i,:], trajectory[i+1,:]))
        mesh_plot.add_lines(traj, color="red", width=40)
                    
    # Plotting final position as a white dot
    trajectory_plot = pv.PolyData(np.transpose(positions[:,-1]))
    mesh_plot.add_mesh(trajectory_plot, color=[1.0, 1.0, 1.0], style='surface')

    mesh_plot.add_axes(x_color='red', y_color='green', z_color='blue', xlabel='X', ylabel='Y', zlabel='Z', line_width=2, shaft_length = 10)
    
    mesh_plot.show(jupyter_backend = 'panel') 

    plt.savefig('figures/67P/trajectory_mesh_plot.png')


def drawSphere(r):
    """ Generates (x,y,z) values for plotting a sphere of given radius r.

    Args:
        r (float): Radius of sphere.

    Returns:
        (x,y,z) (tuple): Arrays of values in cartesian frame, representing a sphere of radius r centered at origin.
    """
    #draw sphere
    u, v = np.mgrid[0:2*np.pi:15j, 0:np.pi:15j] #previously: 40j
    x=r*np.cos(u)*np.sin(v)
    y=r*np.sin(u)*np.sin(v)
    z=r*np.cos(v)

    return (x,y,z)


def plot_UDP(args, positions, plot_mesh, plot_trajectory, plot_risk_zone, view_angle, measurement_radius):
    """plot_trajectory plots the satellite trajectory.
    Args:
        args (dotmap.DotMap): Dotmap dictionary containing info on mesh and bounding spheres.
        positions (np.ndarray): (3xN) Array containing N positions (cartesian frame) along the computed trajectory.
        plot_mesh (bool): Activation of plotting the mesh
        plot_trajectory (bool): Activation of plotting the trajectory
        plot_risk_zone (bool): Activation of plotting the inner bounding sphere (i.e risk-zone)
        view_angle (list): List containing the view angle of the plot.
        measurement_radius (np.ndarray): (N) array containing radius of each measurement sphere at the sampled positions along the trajectory.

    """
    # Define figure
    #ax = plt.figure().add_subplot(projection='3d')
    fig = plt.figure(figsize = (13,7))
    ax = fig.add_subplot(projection='3d')

    #Plot trajectory
    if plot_trajectory:
        x = positions[0,:]
        y = positions[1,:]
        z = positions[2,:]
        ax.plot(x, y, z, label='Trajectory')
        ax.legend()

    # Plot mesh:
    if plot_mesh:
        ax.plot_trisurf(args.mesh.vertices[:, 0], args.mesh.vertices[:,1], triangles=args.mesh.faces, Z=args.mesh.vertices[:,2], alpha=1, color='grey') 

    # Plot risk zone:
    if plot_risk_zone:
        r = args.problem.radius_inner_bounding_sphere
        (x, y, z) = drawSphere(r)
        ax.plot_wireframe(x, y, z, color="r", alpha=0.1)

    # Plot measurement spheres:
    for i in range(0, len(positions[0,:])):
        x_sphere = positions[0,i]
        y_sphere = positions[1,i]
        z_sphere = positions[2,i]
        r_sphere = measurement_radius[i]

        (x_unscaled,y_unscaled,z_unscaled) = drawSphere(r_sphere)
         # shift and scale sphere
        x_scaled = x_unscaled + x_sphere
        y_scaled = y_unscaled + y_sphere
        z_scaled = z_unscaled + z_sphere

        # Plot hollow wireframe:
        #ax.plot_wireframe(xs, ys, zs, color="b", alpha=0.15)

        # Plot surface and wireframe of sphere:
        ax.plot_surface(x_scaled, y_scaled, z_scaled, color="k", alpha=0.9, edgecolor="g")



    # Adjust viewangle of plot:
    ax.view_init(view_angle[0],view_angle[1])

    # Set title of figure:
    ax.set_title("Solution to UDP.    View angle: ("+str(view_angle[0])+", "+str(view_angle[1])+")")

    # Adjust axes limits and figure aspect
    xyzlim = np.array([ax.get_xlim3d(),ax.get_ylim3d(),ax.get_zlim3d()]).T
    XYZlim = [min(xyzlim[0]),max(xyzlim[1])]
    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim)
    ax.set_aspect("equal")

    # Either display figure in jupyter or save to png.
    #plt.show()
    plt.savefig('figures/trajectory_plot.png')