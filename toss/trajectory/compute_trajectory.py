# Import required modules
from toss.mesh.mesh_utility import is_outside
from toss.trajectory.Integrator import IntegrationScheme

# Core packages
import numpy as np
from typing import Union, Callable
import pykep as pk
import desolver as de
import desolver.backend as D
D.set_float_fmt('float64')

def compute_trajectory(x: np.ndarray, args, func: Callable) -> Union[bool, list, np.ndarray]:
    """Computes a single spacecraft trajectory for a given initial state by numerical integation using DEsolver. 

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
                start_time (float): Start time (in seconds) for the integration of trajectory.
                final_time (float): Final time (in seconds) for the integration of trajectory.
                initial_time_step (float): Size of initial time step (in seconds) for integration of trajectory.
                radius_inner_bounding_sphere (float): Radius of the bounding sphere representing risk zone for collisions with celestial body.
                activate_event (bool): Event configuration (0 = no event, 1 = collision with body detection).
                number_of_maneuvers (int): Number of possible maneuvers.
                measurement_period (int): Period for which a measurment sphere is recognized and managed.
            mesh:
                vertices (np.ndarray): Array containing all points on mesh.
                faces (np.ndarray): Array containing all triangles on the mesh.
            state: Parameters provided by the state vector
                time_of_maneuver (float): Time for adding impulsive maneuver [seconds].
                delta_v (np.ndarray): Array containing the cartesian componants of the impulsive maneuver.
        func (Callable): A function handle for the state update equation required for integration.

    Returns:
        collision_penalty (bool): Penalty value given for the event of a collision with the celestial body.
        list_of_trajectory_objects (list): List of OdeSystem integration objects (provided by DEsolver)
        integration_intervals (np.ndarray): (1,N) Array containing the discretized and integrated time intervals. 

    """    
    # Separate initial state from chromosome and translate from osculating elements to cartesian frame.
    r, v = pk.par2ic(E=x[0:6], mu=args.body.mu)
    initial_state = np.array(r+v)

    # In the case of maneuvers:
    if args.problem.number_of_maneuvers > 0:
        integration_intervals, dv_of_maneuvers = setup_maneuvers(x, args)

        #   NOTE: We adjust each time value present in integration_intervals to
        #         be integers since the used solver does not manage floating points
        #         as boundaries for the integrated time interval very well.
        integration_intervals = integration_intervals.astype(np.int32)

        # Rearrange maneuvers into increasing order of time of execution
        maneuver_times = integration_intervals[1:-1]
        correct_order_of_maneuvers = np.argsort(maneuver_times)
        integration_intervals[1:-1] = maneuver_times[correct_order_of_maneuvers]
        dv_of_maneuvers = dv_of_maneuvers[:,correct_order_of_maneuvers]


        # Adjust for simultaneous manuevers
        for idx in range(0,len(integration_intervals)-1):    
            if idx+1 == len(integration_intervals) or idx == len(integration_intervals):
                break
            duplicate_idx = idx
            while integration_intervals[duplicate_idx] == integration_intervals[duplicate_idx+1]:
                dv_of_maneuvers[:,idx] += dv_of_maneuvers[:,duplicate_idx+1]
                duplicate_idx +=1

                if duplicate_idx+1 == len(integration_intervals):
                    break
            
            # Remove duplicate manuver time (and corresponding maneuver)
            integration_intervals = np.delete(integration_intervals, np.arange(idx+1,duplicate_idx+1,1))
            dv_of_maneuvers = np.delete(dv_of_maneuvers, np.arange(idx+1,duplicate_idx+1,1), axis=1)
                
        
    else:
        integration_intervals = np.array([args.problem.start_time, args.problem.final_time])
        integration_intervals = integration_intervals.astype(np.int32)

    # Integrate system for every defined time interval
    list_of_trajectory_objects = []
    final_state = None
    for time_idx in range(0, len(integration_intervals)-1):
        args.integrator.t0 = integration_intervals[time_idx]
        args.integrator.tf = integration_intervals[time_idx+1]

        if time_idx > 0:
            initial_state = final_state
            initial_state[3:6] = dv_of_maneuvers[:, time_idx-1]
        trajectory = integrate_system(func, initial_state, args)
        final_state = trajectory.y[-1, 0:6]
        
        # Save positions with risk-zone entries
        list_of_entry_points = np.empty((len(trajectory.events), 3), dtype=np.float64)
        for array_idx, entry_point in enumerate(trajectory.events):
            list_of_entry_points[array_idx,:] = entry_point.y[0:3]

        if time_idx == 0:
            points_inside_risk_zone = list_of_entry_points
        else:
            points_inside_risk_zone = np.vstack((points_inside_risk_zone, list_of_entry_points))

        # Store OdeSystem trajectory object in list:
        list_of_trajectory_objects.append(trajectory)
        
    # Check for collisions with body
    collisions_avoided = point_is_outside_mesh(points_inside_risk_zone, args.mesh.vertices, args.mesh.faces)
    if all(collisions_avoided) == True:
        collision_detected = False
    else:
        # If collision is detected, stop the integration.
        collision_detected = True

    # Return trajectory and neccessary values for computing fitness in udp.
    return collision_detected, list_of_trajectory_objects, integration_intervals


def setup_maneuvers(x:np.ndarray, args) -> Union[np.ndarray, np.ndarray]:
    """
    Prepares for integration with maneuver by returning one array containing
    each integrated time interval and another with corresponding delta V:s.

    Args:
        x (np.ndarray): State vector.
        args (dotmap.DotMap):
            problem: Parameters related to the problem:
                start_time (float): Start time (in seconds) for the integration of trajectory.
                final_time (float): Final time (in seconds) for the integration of trajectory.
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


def integrate_system(func: Callable, x: np.ndarray, args):
    """ Numerical integration of ODE system using DEsolver library.

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
                initial_time_step (float): Size of initial time step (in seconds) for integration of trajectory.
                radius_inner_bounding_sphere (float): Radius of the bounding sphere representing risk zone for collisions with celestial body.
                activate_event (bool): Event configuration (0 = no event, 1 = collision with body detection)
            mesh:
                vertices (np.ndarray): Array containing all points on mesh.
                faces (np.ndarray): Array containing all triangles on the mesh.
            state: Parameters provided by the state vector
                time_of_maneuver (float): Time for adding impulsive maneuver [seconds].
                delta_v (np.ndarray): Array containing the cartesian componants of the impulsive maneuver.
    Returns:
        ode_object (desolver.differential_system.OdeSystem): The integration object provided by desolver.
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
    
    # Integrate system
    initial_state = D.array(x)
    ode_object = de.OdeSystem(
        func, 
        y0 = initial_state, 
        dense_output = dense_output, 
        t = (t0, tf), 
        dt = dt, 
        rtol = rtol, 
        atol = atol,
        constants=dict(args = args))
    ode_object.method = str(numerical_integrator)

    if activate_event==False:
        ode_object.integrate()

    elif activate_event==True:
        point_is_inside_risk_zone.is_terminal = False
        ode_object.integrate(events=[point_is_inside_risk_zone])

    return ode_object


def point_is_inside_risk_zone(t: float, state: np.ndarray, args) -> int:
    """ Checks for event: Satellite entering inner bounding sphere, that is the risk-zone for body collisions.

    Args:
        t (float): Current time step.
        state (np.ndarray): Current state (position and velocity expressed in cartesian frame)
        args (dotmap.DotMap):
            problem:
                radius_inner_bounding_sphere (float): Radius of bounding sphere around mesh. 

    Returns:
        (int): Returns 0 when the satellite enters the risk-zone, and 1 otherwise.
    """
    risk_zone_radius = args.problem.radius_inner_bounding_sphere
    position = state[0:3]
    distance = risk_zone_radius**2 - (position[0]**2 + position[1]**2 + position[2]**2)
    if distance >= 0:
        return 0
    return 1


def point_is_outside_mesh(x: np.ndarray, mesh_vertices: np.ndarray, mesh_faces: np.ndarray) -> bool:
    """
    Uses is_outside to check if a set of positions (or current) x is inside mesh.

    Args:
        x (np.ndarray): Array containing current, or a set of, positions expressed in 3 dimensions.
        mesh_vertices (np.ndarray): Array containing all points on mesh.
        mesh_faces (np.ndarray): Array containing all triangles on the mesh.

    Returns:
        collision_boolean (bool): Array with boolean values corresponding to each position in x. 
                                  Returns "False" if point is inside mesh, and "True" if point is 
                                  outside mesh (that is, there no collision).
    """
    collision_boolean = is_outside(x, mesh_vertices, mesh_faces)
    return collision_boolean
