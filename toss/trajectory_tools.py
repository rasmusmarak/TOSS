# General
import numpy as np
from typing import Union

# For Plotting
import pyvista as pv

# For working with the mesh
import mesh_utility

# For choosing numerical integration method
from Integrator import IntegrationScheme

# D-solver (performs integration)
import desolver as de
import desolver.backend as D
D.set_float_fmt('float64')


def compute_trajectory(x: np.ndarray, args, func) -> Union[np.ndarray, float, bool]:
    """compute_trajectory computes trajectory of satellite using numerical integation techniques 

    Args:
        x (np.ndarray): State vector containing values for position and velocity of satelite in 3D cartesian coordinates.

    Returns:
        trajectory_info (np.ndarray): Numpy array containing information on position and velocity at every time step (columnwise).
        squared_altitudes (float): Sum of squared altitudes above origin for every position
        collision_penalty (bool): Penalty value given for the event of a collision with the celestial body.
    """
    
    # Compute trajectory by numerical integration
    trajectory, trajectory_info = integrate(func, x, args)

    # Compute average distance to target altitude
    squared_altitudes = trajectory_info[0,:]**2 + trajectory_info[1,:]**2 + trajectory_info[2,:]**2

    # Check for potential collisions
    points_inside_risk_zone = np.empty((len(trajectory.events), 3), dtype=np.float64)
    i = 0
    for j in trajectory.events:
        points_inside_risk_zone[i,:] = j.y[0:3]
        i += 1
    
    collisions_avoided = point_is_outside_mesh(points_inside_risk_zone, args.mesh.vertices, args.mesh.faces)
    if all(collisions_avoided) == True:
        collision_detected = False
    else:
        collision_detected = True
    
    # Return trajectory and neccessary values for computing fitness in udp.
    return trajectory_info, squared_altitudes, collision_detected


def integrate(func, x, args) -> np.ndarray:

    # Setup parameters
    dense_output = args.integrator.dense_output
    t0 = args.problem.start_time
    tf = args.problem.final_time
    dt = args.problem.initial_time_step
    rtol = args.integrator.rtol
    atol = args.integrator.atol
    risk_zone_radius = args.problem.radius_bounding_sphere
    numerical_integrator = IntegrationScheme(args.integrator.algorithm).name
    event = args.problem.event

    # Integrate trajectory
    initial_state = D.array(x)
    trajectory = de.OdeSystem(
        func, 
        y0 = initial_state, 
        dense_output = dense_output, 
        t = (t0, tf), 
        dt = dt, 
        rtol = rtol, 
        atol = atol,
        constants=dict(args = args))
    trajectory.method = str(numerical_integrator)

    if event==0:
        trajectory.integrate()

    elif event==1:
        point_is_inside_risk_zone.is_terminal = False
        trajectory.integrate(events=point_is_inside_risk_zone)

    # Add integration times to trajectory info
    trajectory_info = np.vstack((np.transpose(trajectory.y), trajectory.t))

    return trajectory, trajectory_info


def point_is_inside_risk_zone(t: float, state: np.ndarray, args: float) -> int:
    """ Checks for event: collision with the celestial body.

    Args:
        t (float): Current time step for integration.
        state (np.ndarray): Current state, i.e position and velocity
        risk_zone_radius (float): Radius of bounding sphere around mesh. 

    Returns:
        (int): Returns 1 when the satellite enters the risk-zone, and 0 otherwise.
    """
    risk_zone_radius = args.problem.radius_bounding_sphere
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


def plot_trajectory(r_store: np.ndarray, mesh):
    """plot_trajectory plots the body mesh and satellite trajectory.

    Args:
        r_store (np.ndarray): Array containing values on position at each time step for the trajectory (columnwise).
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