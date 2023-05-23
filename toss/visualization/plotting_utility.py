 # core stuff
import numpy as np

# For Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv

def distance_deviation_over_time(n_spacecraft, r_bound_min, r_bound_max, positions, timelist):
    """ Plots the distance deviation over time
    """
    # Plot fitness over generations
    figure, ax = plt.subplots(figsize=(5, 5))

    if n_spacecraft > 1:
        positions_list = np.array_split(positions, n_spacecraft, axis=1)
        for spacecraft_idx in range(0, len(positions_list)):
            pos = positions_list[spacecraft_idx]
            distances = (pos[0,:]**2 + pos[1,:]**2 + pos[2,:]**2)**(1/2)
            ax.plot(timelist, distances, label='Trajectory '+str(spacecraft_idx+1))
    else:
        distances = (positions[0,:]**2 + positions[1,:]**2 + positions[2,:]**2)**(1/2)
        ax.plot(timelist, distances, label='Trajectory')

    # plot inner bounding sphere radius
    ax.axhline(y = r_bound_min, color = 'r', linestyle = 'dashed', label = "Inner bounding sphere")

    # Plot outer bounding sphere radius
    ax.axhline(y = r_bound_max, color = 'r', linestyle = ':', label = "Outer bounding sphere")

    # Change ticks aspect ratio from meters to kilometers
    m2km = lambda y, _: f'{y/1000:g}'
    ax.yaxis.set_major_formatter(m2km)

    s2days = lambda x, _: f'{int((x/3600)/24):g}'
    ax.xaxis.set_major_formatter(s2days)

    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Distance [km]')

    # Adjust plot settings
    ax.grid('major')
    #ax.set_title('Distance deviation over time', fontweight='bold')
    ax.legend(loc='upper left')
    #ax.set_yscale('log')
    plt.tight_layout()
    #plt.show()
    plt.savefig('distance_deviation_over_time.png')


def fitness_over_time(coverage_list, close_dist_penalty_list, far_dist_penalty_list, time_list):
    """ Plots the fitness value evolution over time
    Args:
        fitness_list (np.ndarray): Array containing champion fitness vlue of each generation.
        timelist (list): List of time values
    """
    # Plot fitness over generations
    figure, ax = plt.subplots(figsize=(5, 5))
    ax.plot(time_list, coverage_list, label='Coverage')
    ax.plot(time_list, close_dist_penalty_list, label='Close Distance Penalty')
    ax.plot(time_list, far_dist_penalty_list, label='Far Distance Penalty')

    # Adjust plot settings
    #ax.set_xlim((0, timelist))
    ax.grid('major')
    #ax.set_title('Coverage evolution', fontweight='bold')
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Fitness')
    ax.legend(loc='upper left')
    #ax.set_yscale('log')
    plt.tight_layout()
    #plt.show()
    plt.savefig('coverage_over_time.png')

def fitness_over_generations(fitness_list, number_of_generations):
    """ Plots the champion fitness vlue of each generation
    Args:
        fitness_list (np.ndarray): Array containing champion fitness vlue of each generation.
        number_of_generations (int): Number of generations for the optimization.
    """
    # Plot fitness over generations
    figure, ax = plt.subplots(figsize=(5, 5))
    ax.plot(np.arange(0, number_of_generations), fitness_list, label='Fitness function')

    # Find fist occurance of true champion
    #champion = np.min(fitness_list)
    #champion_idx = np.where(fitness_list==champion)[0][0]
    #ax.scatter(champion_idx, champion, marker='x', color='r', label='All-time champion \n[Gen: '+str(champion_idx)+"]")

    # Find fist occurance of approximate convergence (compared to true champion with given error margin)
    #convergence_list = fitness_list - champion
    #convergence_idx = np.where(convergence_list < 1e-6)[0][0]
    #convergence_value = fitness_list[convergence_idx]
    #print(convergence_idx)
    #ax.scatter(convergence_idx, convergence_value, marker='x', color='g', label='Convergence (delta < 1e-6) \n[Gen: '+str(convergence_idx)+"]")

    # Adjust plot settings
    ax.set_xlim((0, number_of_generations))
    ax.grid('major')
    #ax.set_title('Best individual of each generation', fontweight='bold')
    ax.set_xlabel('Number of generation')
    ax.set_ylabel('UDP fitness value')
    ax.legend(loc='upper right')
    ax.set_yscale('log')
    plt.tight_layout()
    #plt.show()
    plt.savefig('fitness_over_generations.png')

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
    ax2.set_title("Axis: (x,z)")

    ax3.plot(trajectory_info[2,:],trajectory_info[3,:])
    ax3.set_title("Axis: (y,z)")

    plt.savefig('figures/two_axis_plot.png')

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


def plot_UDP(args, positions, maneuver_positions, maneuver_unit_vectors, plot_mesh, plot_trajectory, plot_control, plot_risk_zone, plot_measurement_sphere, view_angle, measurement_radius):
    """plot_trajectory plots the satellite trajectory.
    Args:
        args (dotmap.DotMap): Dotmap dictionary containing info on mesh and bounding spheres.
        positions (np.ndarray): (3xN) Array containing N positions (cartesian frame) along the computed trajectory.
        maneuver_positions (np.ndarray): (3xN) Array containing N positions (cartesian frame) corresponding to control vectors.
        maneuver_unit_vectors (np.ndarray): (3xN) Array containing N control vectors (cartesian frame) along the computed trajectory.
        plot_mesh (bool): Activation of plotting the mesh
        plot_trajectory (bool): Activation of plotting the trajectory
        plot_control (bool): Activation of plotting the control vectors (direction of thrust)
        plot_risk_zone (bool): Activation of plotting the inner bounding sphere (i.e risk-zone)
        plot_measurement_sphere (bool): Activation of plotting measurement spheres.
        view_angle (list): List containing the view angle of the plot.
        measurement_radius (np.ndarray): (N) array containing radius of each measurement sphere at the sampled positions along the trajectory.
    """
    # Define figure
    #ax = plt.figure().add_subplot(projection='3d')
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(projection='3d')

    #Plot trajectory
    if plot_trajectory:
        id = 1
        n_pos = int((args.problem.final_time/args.problem.measurement_period))
        for i in range(0,len(positions[0,:]), n_pos):
            x = positions[0, i:(i+n_pos)]
            y = positions[1, i:(i+n_pos)]
            z = positions[2, i:(i+n_pos)]

            # Plot square marker at starting position
            ax.scatter(x[0], y[0], z[0], marker='s', c='black')

            # Plot pointer at final position
            ax.scatter(x[-1], y[-1], z[-1], marker='o', c='black')

            # Plot trajectory
            ax.plot(x, y, z, label='Trajectory '+str(id))
            ax.legend()
            id += 1

    # Plotting control 
    if plot_control:
        for maneuver_idx in range(0, len(maneuver_unit_vectors[0,:])):
            x = maneuver_positions[0,maneuver_idx]
            y = maneuver_positions[1,maneuver_idx]
            z = maneuver_positions[2,maneuver_idx]
            ux = maneuver_unit_vectors[0,maneuver_idx]
            uy = maneuver_unit_vectors[1,maneuver_idx]
            uz = maneuver_unit_vectors[2,maneuver_idx]
            ax.quiver(x, y, z, -ux, -uy, -uz, color='r', length=2500) #arrow_length_ratio
        
    # Plot mesh:
    if plot_mesh:
        ax.plot_trisurf(args.mesh.vertices[:, 0], args.mesh.vertices[:,1], triangles=args.mesh.faces, Z=args.mesh.vertices[:,2], alpha=1, color='grey') 

    # Plot risk zone:
    if plot_risk_zone:
        r = args.problem.radius_inner_bounding_sphere
        (x, y, z) = drawSphere(r)
        ax.plot_wireframe(x, y, z, color="r", alpha=0.1)

    # Plot measurement spheres:
    if plot_measurement_sphere:
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
    #ax.set_title("Solution to UDP.    View angle: ("+str(view_angle[0])+", "+str(view_angle[1])+")")

    # Adjust axes limits and figure aspect
    xyzlim = np.array([ax.get_xlim3d(),ax.get_ylim3d(),ax.get_zlim3d()]).T
    XYZlim = [min(xyzlim[0]),max(xyzlim[1])]
    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim)

    ax.set_aspect("equal")

    # Change ticks aspect ratio from meters to kilometers
    m2km = lambda x, _: f'{x/1000:g}'
    ax.xaxis.set_major_formatter(m2km)
    ax.yaxis.set_major_formatter(m2km)
    ax.zaxis.set_major_formatter(m2km)

    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")

    # Either display figure in jupyter or save to png.
    #plt.show()
    plt.savefig('trajectory_plot.png') # fixed_body_frame_trajectory_plot # trajectory_plot

def plot_performance_scaling(core_counts, run_times):

    # Setting up figure
    figure, (ax1, ax2) = plt.subplots(1,2)
    figure.tight_layout()

    # Define ideal scaling:
    lowest_core_count = core_counts[0]
    measured_time_lowest_core_count = run_times[0]
    ideal_time_n_cores=[]
    speed_up = []
    for i in range(0,len(core_counts)):
        n_cores = core_counts[i]
        ideal_time_n_cores.append((measured_time_lowest_core_count * lowest_core_count)/n_cores)
        speed_up.append(measured_time_lowest_core_count/run_times[i])

    # Plotting: run time vs core count (alongside corresponding ideal scaling)
    ax1.plot(core_counts, run_times, 'o-r') #, label="10 Pop/Island"
    ax1.plot(core_counts, ideal_time_n_cores, "--b", label="Ideal scaling")
    ax1.legend()
    ax1.set_xlabel("Number of workers in pool")
    ax1.set_ylabel("Run time (Seconds)") 
    ax1.set_xscale("log")
    ax1.set_yscale("log")

    # Plotting: efficiency vs core count (and a reference line representing 80%)
    efficiency = []
    for j in range(0,len(core_counts)):
        efficiency.append(ideal_time_n_cores[j]/run_times[j])
    ax2.plot(core_counts, efficiency, 'o-r') #, label="10 Pop/Island"
    ax2.axhline(y=0.8, color="b", linestyle="--", label="80% Reference")
    ax2.plot(core_counts, speed_up, "--g", label="Speed-up")


    ax2.legend()
    ax2.set_xlabel("Number of workers in pool")
    ax2.set_ylabel("Efficiency")
    ax2.set_xscale("log")
    ax2.set_yscale("log")