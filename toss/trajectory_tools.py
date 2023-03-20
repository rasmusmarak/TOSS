# General
import numpy as np
from typing import Union, Callable

# For orbit representation (reference frame)
import pykep as pk

# For working with the mesh
from toss.mesh_utility import is_outside

# For choosing numerical integration method
from toss.Integrator import IntegrationScheme

# D-solver (performs integration)
import desolver as de
import desolver.backend as D
D.set_float_fmt('float64')


def compute_trajectory(x: np.ndarray, args, func: Callable) -> Union[np.ndarray, float, bool]:
    """compute_trajectory computes trajectory of satellite using numerical integation techniques 

    Args:
        x (np.ndarray): State vector.
        args (dotmap.DotMap):
            body: Parameters related to the celestial body:
                density (float): Mass density of celestial body.
                mu (float): Gravitational parameter for celestial body.
                declination (float): Declination angle of spin axis.
                right_ascension (float): Right ascension angle of spin axis.
                spin_velocity (float): Angular velocity of the body's rotation.
                spin_axis (np.ndarray): The axis around which the body rotates.
            integrator: Specific parameters related to the integrator:
                algorithm (int): Integer representing specific integrator algorithm.
                dense_output (bool): Dense output status of integrator.
                rtol (float): Relative error tolerance for integration.
                atol (float): Absolute error tolerance for integration.
            problem: Parameters related to the problem:
                start_time (int): Start time (in seconds) for the integration of trajectory.
                final_time (int): Final time (in seconds) for the integration of trajectory.
                initial_time_step (float): Size of initial time step (in seconds) for integration of trajectory.
                radius_bounding_sphere (float): Radius of the bounding sphere representing risk zone for collisions with celestial body.
                activate_event (bool): Event configuration (0 = no event, 1 = collision with body detection).
                number_of_maneuvers (int): Number of possible maneuvers.
            mesh:
                vertices (np.ndarray): Array containing all points on mesh.
                faces (np.ndarray): Array containing all triangles on the mesh.
            state: Parameters provided by the state vector
                time_of_maneuver (float): Time for adding impulsive maneuver [seconds].
                delta_v (np.ndarray): Array containing the cartesian componants of the impulsive maneuver.
        func (Callable): A function handle for the state update equation required for integration.

    Returns:
        trajectory_info (np.ndarray): Numpy array containing information on position and velocity at every time step (columnwise).
        squared_altitudes (float): Sum of squared altitudes above origin for every position
        collision_penalty (bool): Penalty value given for the event of a collision with the celestial body.
    """
    # Separate initial state from chromosome and translate from osculating elements to cartesian frame.
    r, v = pk.par2ic(E=x[0:6], mu=args.body.mu)
    initial_state = np.array(r+v)

    # In the case of maneuvers:
    if args.problem.number_of_maneuvers > 0:
        integration_intervals, dv_of_maneuvers = setup_maneuvers(x, args)
        integration_intervals = integration_intervals.astype(np.int32)

        # Rearrange maneuvers into increasing order of time of execution
        maneuver_times = integration_intervals[1:-1]
        correct_order_of_maneuvers = np.argsort(maneuver_times)
        integration_intervals[1:-1] = maneuver_times[correct_order_of_maneuvers]
        dv_of_maneuvers = dv_of_maneuvers[:,correct_order_of_maneuvers]
        
    else:
        integration_intervals = [args.problem.start_time, args.problem.final_time]

    # Integrate system for every defined time interval
    trajectory_info = None
    for time_idx in range(0, len(integration_intervals)-1):
        args.integrator.t0 = integration_intervals[time_idx]
        args.integrator.tf = integration_intervals[time_idx+1]

        #print("Time interval: [", args.integrator.t0, args.integrator.tf, "].")

        if time_idx == 0: # First integrated time-interval.
            trajectory, trajectory_memory = integrate_trajectory(func, initial_state, args)
            trajectory_info = trajectory_memory

        else:
            initial_state = trajectory_info[0:6,-1]
            initial_state[3:6] = dv_of_maneuvers[:, time_idx-1]
            trajectory, trajectory_memory = integrate_trajectory(func, initial_state, args)
            trajectory_info = np.hstack((trajectory_info[:,0:-1] ,trajectory_memory))
        
        # Save positions with risk-zone entries
        list_of_entry_points = np.empty((len(trajectory.events), 3), dtype=np.float64)
        array_idx = 0
        for entry_point in trajectory.events:
            list_of_entry_points[array_idx,:] = entry_point.y[0:3]
            array_idx += 1

        if time_idx == 0: 
            points_inside_risk_zone = list_of_entry_points
        else:
            points_inside_risk_zone = np.vstack((points_inside_risk_zone, list_of_entry_points))


    # Compute average distance to target altitude
    squared_altitudes = trajectory_info[0,:]**2 + trajectory_info[1,:]**2 + trajectory_info[2,:]**2
    
    # Check for collisions with body
    collisions_avoided = point_is_outside_mesh(points_inside_risk_zone, args.mesh.vertices, args.mesh.faces)
    if all(collisions_avoided) == True:
        collision_detected = False
    else:
        collision_detected = True
    
    # Return trajectory and neccessary values for computing fitness in udp.
    return trajectory_info, squared_altitudes, collision_detected


def setup_maneuvers(x:np.ndarray, args) -> Union[np.ndarray, np.ndarray]:
    """
    Prepares for integration with maneuver by returning one array containing
    each integrated time interval and another with corresponding delta V:s.

    Args:
        x (np.ndarray): State vector.
        args (dotmap.DotMap):
            problem: Parameters related to the problem:
                start_time (int): Start time (in seconds) for the integration of trajectory.
                final_time (int): Final time (in seconds) for the integration of trajectory.
                number_of_maneuvers (int): Number of possible maneuvers.

    Returns:
        integration_intervals (np.ndarray): Array of discretized time steps for the integration.
        dv_of_maneuvers (np.ndarray): Array of the delta v corresponding to each maneuver.
    """

    # Separate maneuvers from chromosome
    list_of_maneuvers = np.array_split(np.array(x[6:]), len(x[6:])/4)

    # Setup time and dv arrays for discretized integration
    integration_intervals = np.empty((args.problem.number_of_maneuvers+2), dtype=np.float64)
    dv_of_maneuvers = np.empty((3, args.problem.number_of_maneuvers), dtype=np.float64)

    # Add start and final time of integration
    integration_intervals[0] = args.problem.start_time
    integration_intervals[-1] = args.problem.final_time

    # Store time and dv of each maneuver in separate arrays
    idx = 0 #index
    for maneuver in list_of_maneuvers:
        integration_intervals[idx+1] = maneuver[0]
        dv_of_maneuvers[:,idx] = maneuver[1:]
        idx += 1
    
    return integration_intervals, dv_of_maneuvers


def integrate_trajectory(func: Callable, x: np.ndarray, args):

    """ Integrates trajectory numerically using DeSolver library.

    Args:
        func (Callable): A function handle for the equ_rhs (state update equation) required for integration.
        x (np.ndarray): Initial position and velocity of satelite in 3D cartesian coordinates.
        args (dotmap.DotMap):
            body:
                density (float): Mass density of celestial body.
            integrator: Specific parameters related to the integrator:
                algorithm (int): Integer representing specific integrator algorithm.
                dense_output (bool): Dense output status of integrator.
                rtol (float): Relative error tolerance for integration.
                atol (float): Absolute error tolerance for integration.
            problem: Parameters related to the problem:
                start_time (int): Start time (in seconds) for the integration of trajectory.
                final_time (int): Final time (in seconds) for the integration of trajectory.
                initial_time_step (float): Size of initial time step (in seconds) for integration of trajectory.
                radius_bounding_sphere (float): Radius of the bounding sphere representing risk zone for collisions with celestial body.
                activate_event (bool): Event configuration (0 = no event, 1 = collision with body detection)
            mesh:
                vertices (np.ndarray): Array containing all points on mesh.
                faces (np.ndarray): Array containing all triangles on the mesh.
            state: Parameters provided by the state vector
                time_of_maneuver (float): Time for adding impulsive maneuver [seconds].
                delta_v (np.ndarray): Array containing the cartesian componants of the impulsive maneuver.
    Returns:
        trajectory (desolver.differential_system.OdeSystem): The integration object provided by desolver.
        trajectory_info (np.ndarray): Numpy array containing information on position and velocity at every time step (columnwise).
    """

    # Setup parameters
    dense_output = args.integrator.dense_output
    t0 = args.integrator.t0
    tf = args.integrator.tf
    dt = args.problem.initial_time_step
    rtol = args.integrator.rtol
    atol = args.integrator.atol
    numerical_integrator = IntegrationScheme(args.integrator.algorithm).name
    activate_event = args.problem.activate_event

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

    if activate_event==False:
        trajectory.integrate()

    elif activate_event==True:
        point_is_inside_risk_zone.is_terminal = False
        trajectory.integrate(events=point_is_inside_risk_zone)
    
    # Add integration times to trajectory info
    trajectory_info = np.vstack((np.transpose(trajectory.y), trajectory.t))

    return trajectory, trajectory_info


def point_is_inside_risk_zone(t: float, state: np.ndarray, args) -> int:
    """ Checks for event: collision with the celestial body.

    Args:
        t (float): Current time step for integration.
        state (np.ndarray): Current state, i.e position and velocity
        args (dotmap.DotMap):
            problem:
                radius_bounding_sphere (float): Radius of bounding sphere around mesh. 

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
        mesh_vertices (np.ndarray): Array containing all points on mesh.
        mesh_faces (np.ndarray): Array containing all triangles on the mesh.

    Returns:
        collision_boolean (bool): A one dimensional array with boolean values corresponding to each
                                position kept in x. Returns "False" if point is inside mesh, and 
                                "True" if point is outside mesh (that is, there no collision).
    """
    collision_boolean = is_outside(x, mesh_vertices, mesh_faces)
    return collision_boolean
