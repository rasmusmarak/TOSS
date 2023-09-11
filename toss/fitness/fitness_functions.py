# Core packages
import numpy as np
from typing import Union
from toss.fitness.fitness_function_enums import FitnessFunctions
from toss.fitness.fitness_function_utils import _compute_squared_distance, estimate_covered_volume, compute_space_coverage

def get_fitness(chosen_fitness_function: FitnessFunctions, args, positions: np.ndarray, velocities:np.ndarray, timesteps: np.ndarray):
    """ Returns user specified fitness function.
    Args:
        chosen_function (FitnessFunctions): Chosen fitness function as defined in enum class FitnessFunctions.
        args (dotmap.DotMap): Dotmap with the required values for each defined fitness function.
        positions (np.ndarray): (3,N) Array of positions along the trajectory.
        velocities (np.ndarray): (3,N) Array of velocities along the trajectory.
        timesteps (np.ndarray): (N) Array of time values for each position.

    Returns:
        (float): The evaluated fitness value correspondning to chosen_function.
    """
    
    if chosen_fitness_function == FitnessFunctions.TargetAltitudeDistance:
        return target_altitude_distance(args.problem.target_squared_altitude, positions)
    
    elif chosen_fitness_function == FitnessFunctions.CloseDistancePenalty:
        return close_distance_penalty(args.problem.radius_inner_bounding_sphere, positions, args.problem.penalty_scaling_factor)
    
    elif chosen_fitness_function == FitnessFunctions.FarDistancePenalty:
        return far_distance_penalty(args.problem.radius_outer_bounding_sphere, positions, args.problem.penalty_scaling_factor)
    
    elif chosen_fitness_function == FitnessFunctions.CoveredVolume:
        return covered_volume(args.problem.maximal_measurement_sphere_volume, positions)

    elif chosen_fitness_function == FitnessFunctions.TotalCoveredVolume:
        return total_covered_volume(args.problem.total_measurable_volume, positions)

    elif chosen_fitness_function == FitnessFunctions.CoveredVolumeFarDistancePenalty:
        return covered_volume_far_distance_penalty(args.problem.maximal_measurement_sphere_volume, args.problem.radius_outer_bounding_sphere, positions, args.problem.penalty_scaling_factor)
    
    elif chosen_fitness_function == FitnessFunctions.CoveredVolumeCloseDistancePenaltyFarDistancePenalty:
        return covered_volume_close_distance_penalty_far_distance_penalty(args.problem.maximal_measurement_sphere_volume, args.problem.radius_inner_bounding_sphere, args.problem.radius_outer_bounding_sphere, positions, args.problem.penalty_scaling_factor)
    
    elif chosen_fitness_function == FitnessFunctions.CoveredSpace:
        return covered_space(args.problem.number_of_spacecrafts, args.body.spin_axis, args.body.spin_velocity, args.problem.radius_inner_bounding_sphere, args.problem.radius_outer_bounding_sphere, positions, velocities, timesteps, args.problem.tensor_grid_r, args.problem.tensor_grid_theta, args.problem.tensor_grid_phi, args.problem.weight_tensor, args.body.quaternion_rotation_objects)
    
    elif chosen_fitness_function == FitnessFunctions.CoveredSpaceCloseDistancePenaltyFarDistancePenalty:
        return covered_space_close_distance_penalty_far_distance_penalty(args.problem.number_of_spacecrafts, args.body.spin_axis, args.body.spin_velocity, args.problem.radius_inner_bounding_sphere, args.problem.radius_outer_bounding_sphere, positions, velocities, timesteps, args.problem.penalty_scaling_factor, args.problem.tensor_grid_r, args.problem.tensor_grid_theta, args.problem.tensor_grid_phi, args.problem.weight_tensor, args.body.quaternion_rotation_objects)


def target_altitude_distance(target_squared_altitude: float, positions: np.ndarray) -> float:
    """ Computes average distance to target altitude for satellite positions along the trajectory.
    Args:
        target_squared_altitude (float): Squared value of target altitude (from origin in cartesian frame)
        positions (np.ndarray): (3,N) Array of positions along the trajectory.

    Returns:
        fitness (float): Average distance to target altitude.
    """    
    average__squared_deviation_distance = np.mean(np.abs(_compute_squared_distance(positions, (target_squared_altitude**(1/2)))))
    fitness = (average__squared_deviation_distance/target_squared_altitude)**(1/4)
    return fitness


def close_distance_penalty(radius_inner_bounding_sphere: float, positions: np.ndarray, penalty_scaling_factor: float) -> float:
    """ Computes penalty depending on the position closest to the body and inside risk-zone.
    Args:
        radius_inner_bounding_sphere (float): Radius of inner bounding sphere.
        positions (np.ndarray): (3,N) Array of positions along the trajectory.
        penalty_scaling_factor (float): A factor to rescale the penalty.

    Returns:
        Penalty (float): Penalty defined between [0,1].
    """
    # For each point along the trajectory, compute squared distance to inner sphere radius
    delta_distance_squared = _compute_squared_distance(positions, radius_inner_bounding_sphere)

    # We only want to penalize positions that are inside inner-sphere (i.e risk-zone).
    delta_distance_squared = delta_distance_squared[delta_distance_squared<0]
    if len(delta_distance_squared) == 0:
        return 0
    else:
        # For positions inside sphere, identify the one farthest away from radius (i.e with greatest risk)
        maximum_distance = np.abs(np.min(delta_distance_squared))
        
        # Determine penalty P=[0,1]*penalty_scaling_factor depending on distance. 
        penalty = ((maximum_distance/(radius_inner_bounding_sphere**2))**(1/4)) * penalty_scaling_factor
        return penalty


def far_distance_penalty(radius_outer_bounding_sphere: float, positions: np.ndarray, penalty_scaling_factor: float) -> float:
    """ Computes penalty depending on the mean distance to the measurement-zone.
    Args:
        radius_outer_bounding_sphere (float): Radius of outer bounding sphere.
        positions (np.ndarray): (3,N) Array of positions along the trajectory.
        penalty_scaling_factor (float): A factor to rescale the penalty.

    Returns:
        Penalty (float): Penalty defined between [0,1]*penalty_scaling_factor.
    """
    # For each point along the trajectory, compare the squared distance of the position with to the outer sphere radius
    delta_distance_squared = _compute_squared_distance(positions, radius_outer_bounding_sphere)

    # We only want to penalize positions that are outside outer-sphere (i.e measurement-zone).
    # That is whenever the distance from origin to the position is greater than the radius of the
    # outer bounding sphere.
    delta_distance_squared = delta_distance_squared[delta_distance_squared>0]
    if len(delta_distance_squared)==0:
        return 0
    else:
        # For positions outside sphere, identify the mean distande from radius
        mean_distance = np.mean(delta_distance_squared)
        
        # Determine penalty depending on mean distance. 
        penalty = ((mean_distance/(radius_outer_bounding_sphere**2))**(1/4)) * penalty_scaling_factor
        return penalty


def covered_volume(maximal_measurement_sphere_volume: float, positions: np.ndarray) -> float:
    """Computes the ration of unmeasured volume.
    Args:
        maximal_measurement_sphere_volume (float): Total measurable volume for a measurement sphere at the inner bounding sphere radius positioned at estimated greatest gravitational influence from the body.
        positions (np.ndarray): (3,N) Array of positions along the trajectory.

    Returns:
        measured_volume_ratio (float): Ratio of unmeasured volume.
    """
    _, estimated_volume = estimate_covered_volume(positions)
    fitness = -(estimated_volume/(maximal_measurement_sphere_volume*len(positions[0,:])))
    return fitness

def total_covered_volume(total_measurable_volume: float, positions: np.ndarray) -> float:
    """
    Computes the ratio of measured volume iniside the search space to
    the total measurable volume defined by the measurement of a satellite
    positioned at the inner-bounding-sphere radius in close proximity to a point
    near the body's greatest gravitational influence.

    Args:
        maximal_measurement_sphere_volume (float): Total measurable volume for a measurement sphere at the inner bounding sphere radius positioned at estimated greatest gravitational influence from the body.
        positions (np.ndarray): (3,N) Array of positions along the trajectory.

    Returns:
        measured_volume_ratio (float): Ratio of unmeasured volume.
    """
    _, estimated_volume = estimate_covered_volume(positions)
    fitness = -(estimated_volume/total_measurable_volume)
    return fitness

def covered_volume_far_distance_penalty(maximal_measurement_sphere_volume: float, radius_outer_bounding_sphere: float, positions: np.ndarray, penalty_scaling_factor:float) -> float:
    """ Returns combined fitness value of CoveredVolume and FarDistancePenalty
    Args:
        maximal_measurement_sphere_volume (float): Total measurable volume for a measurement sphere at the inner bounding sphere radius positioned at estimated greatest gravitational influence from the body.
        radius_outer_bounding_sphere (float): Radius of outer bounding sphere.
        positions (np.ndarray): (3,N) Array of positions along the trajectory.
        penalty_scaling_factor (float): A factor to rescale the penalty.

    Returns:
        (float): Aggregate fitness value.
    """
    return (covered_volume(maximal_measurement_sphere_volume,positions) + far_distance_penalty(radius_outer_bounding_sphere,positions,penalty_scaling_factor))


def covered_volume_close_distance_penalty_far_distance_penalty(maximal_measurement_sphere_volume: float, radius_inner_bounding_sphere: float, radius_outer_bounding_sphere: float, positions: np.ndarray, penalty_scaling_factor):
    """ Returns aggregate fitness of covered_volume, close_distance_penalty and far_distance_penalty.
    Args:
        maximal_measurement_sphere_volume (float): Total measurable volume for a measurement sphere at the inner bounding sphere radius positioned at estimated greatest gravitational influence from the body.
        radius_inner_bounding_sphere (float): Radius of inner bounding sphere.
        radius_outer_bounding_sphere (float): Radius of outer bounding sphere.
        positions (np.ndarray): (3,N) Array of positions along the trajectory.
        penalty_scaling_factor (float): A factor to rescale the penalty.

    Returns:
        (float): Aggregate fitness value.
    """
    return (covered_volume(maximal_measurement_sphere_volume,positions) + close_distance_penalty(radius_inner_bounding_sphere, positions, penalty_scaling_factor) + far_distance_penalty(radius_outer_bounding_sphere,positions, penalty_scaling_factor))

def covered_space(number_of_spacecrafts: int, spin_axis: np.ndarray, spin_velocity: float, radius_inner_bounding_sphere: float, radius_outer_bounding_sphere: float, positions: np.ndarray, velocities: np.ndarray, timesteps: np.ndarray, r: np.ndarray, theta: np.ndarray, phi: np.ndarray, weight_tensor: np.ndarray, quaternion_objects: np.ndarray or None) -> float:
    """ Returns the ratio of visited points to a number of points definied inside the outer bounding sphere.

    Args:
        number_of_spacecrafts (int): Number of spacecraft
        spin_axis (np.ndarray): The axis around which the body rotates.
        spin_velocity (float): Angular velocity of the body's rotation.
        radius_inner_bounding_sphere (float):  Radius of inner bounding sphere.
        radius_outer_bounding_sphere (float):  Radius of outer bounding sphere.
        positions (np.ndarray): (3,N) Array of positions along the trajectory.
        velocities (np.ndarray): (3,N) Array of velocities along the trajectory.
        timesteps (np.ndarray): (N) Array of time values for each position.
        r (np.ndarray): Array of r coordinates for each point defined on the spherical tensor.
        theta (np.ndarray): Array of theta coordinates for each point defined on the spherical tensor.
        phi (np.ndarray): Array of phi coordinates for each point defined on the spherical tensor.
        weight_tensor (np.ndarray): Array of normalized weights corresponding to each point defined on the spherical tensor.
        quaternion_objects (np.ndarray or None): Prepared quaternion objects for rotating complete trajetcory using predetermined angles.

    Returns:
        visited_space_ratio (float): ratio of visited points to a number of points definied inside the outer bounding sphere.
    """
    visited_space_ratio = compute_space_coverage(number_of_spacecrafts, spin_axis, spin_velocity, positions, velocities, timesteps, radius_inner_bounding_sphere, radius_outer_bounding_sphere, r, theta, phi, weight_tensor, quaternion_objects)
    return visited_space_ratio

def covered_space_close_distance_penalty_far_distance_penalty(number_of_spacecrafts: int, spin_axis: np.ndarray, spin_velocity: float, radius_inner_bounding_sphere: float, radius_outer_bounding_sphere: float, positions: np.ndarray, velocities: np.ndarray, timesteps: np.ndarray, penalty_scaling_factor: float, r: np.ndarray, theta: np.ndarray, phi: np.ndarray, weight_tensor: np.ndarray, quaternion_objects: np.ndarray or None) -> float:
    """ Returns aggregate fitness of covered_space, close_distance_penalty and far_distance_penalty.

    Args:
        number_of_spacecrafts (int): Number of spacecraft
        spin_axis (np.ndarray): The axis around which the body rotates.
        spin_velocity (float): Angular velocity of the body's rotation.
        radius_inner_bounding_sphere (float): Radius of inner bounding sphere.
        radius_outer_bounding_sphere (float): Radius of outer bounding sphere.
        positions (np.ndarray): (3,N) Array of positions along the trajectory.
        velocities (np.ndarray): (3,N) Array of velocities along the trajectory.
        timesteps (np.ndarray): (N) Array of time values for each position.
        penalty_scaling_factor (float): A factor to rescale the penalty.
        r (np.ndarray): Array of r coordinates for each point defined on the spherical tensor.
        theta (np.ndarray): Array of theta coordinates for each point defined on the spherical tensor.
        phi (np.ndarray): Array of phi coordinates for each point defined on the spherical tensor.
        weight_tensor (np.ndarray): Array of normalized weights corresponding to each point defined on the spherical tensor.
        quaternion_objects (np.ndarray or None): Prepared quaternion objects for rotating complete trajetcory using predetermined angles.

    Returns:
        fitness (float): Aggregate fitness value.
    """

    # Compute coverage jointly
    coverage = covered_space(number_of_spacecrafts, spin_axis, spin_velocity, radius_inner_bounding_sphere, radius_outer_bounding_sphere, positions, velocities, timesteps, r, theta, phi, weight_tensor, quaternion_objects)
    closedistancepenalty = close_distance_penalty(radius_inner_bounding_sphere, positions, penalty_scaling_factor)
    fardistancepenalty = far_distance_penalty(radius_outer_bounding_sphere,positions,penalty_scaling_factor)

    # Compute fitness    
    fitness = closedistancepenalty + fardistancepenalty - coverage
    return fitness