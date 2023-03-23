# General
import numpy as np
from typing import Union, Callable
 
# For working with the mesh
from mesh_utility import is_outside

# For choosing numerical integration method
from Integrator import IntegrationScheme

# D-solver (performs integration)
import desolver as de
import desolver.backend as D
D.set_float_fmt('float64')



def compute_trajectory(x: np.ndarray, args, func: Callable) -> Union[np.ndarray, float, bool]:
    """compute_trajectory computes trajectory of satellite using numerical integation techniques 

    Args:
        x (np.ndarray): State vector containing values for position and velocity of satelite in 3D cartesian coordinates.
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
                radius_bounding_sphere (float): Radius of the bounding sphere representing risk zone for collisions with celestial body.
                event (int): Event configuration (0 = no event, 1 = collision with body detection)
        func (Callable): A function handle for the state update equation required for integration.

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


def integrate(func: Callable, x: np.ndarray, args):

    """ Integrates trajectory numerically using DeSolver library.

    Args:
        func (Callable): A function handle for the equ_rhs (state update equation) required for integration.
        x (np.ndarray): State vector containing values for position and velocity of satelite in 3D cartesian coordinates.
        args (dotmap.DotMap):
            body:
                density (float): Mass density of celestial body.
            integrator: Specific parameters related to the integrator:
                algorithm (int): Integer representing specific integrator algorithm.
                dense_output (bool): Dense output status of integrator.
                rtol (float): Relative error tolerance for integration.
                atol (float): Absolute error tolerance for integration.
            problem: Parameters related to the problem:
                start_time (float): Start time (in seconds) for the integration of trajectory.
                final_time (float): Final time (in seconds) for the integration of trajectory.
                initial_time_step (float): Size of initial time step (in seconds) for integration of trajectory.
                radius_bounding_sphere (float): Radius of the bounding sphere representing risk zone for collisions with celestial body.
                event (int): Event configuration (0 = no event, 1 = collision with body detection)
            mesh:
                vertices (np.ndarray): Array containing all points on mesh.
                faces (np.ndarray): Array containing all triangles on the mesh.
    Returns:
        trajectory (desolver.differential_system.OdeSystem): The integration object provided by desolver.
        trajectory_info (np.ndarray): Numpy array containing information on position and velocity at every time step (columnwise).
    """

    # Setup parameters
    dense_output = args.integrator.dense_output
    t0 = args.problem.start_time
    tf = args.problem.final_time
    dt = args.problem.initial_time_step
    rtol = args.integrator.rtol
    atol = args.integrator.atol
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
