# Core packages
import numpy as np
import typing
from toss.fitness.fitness_function_enums import FitnessFunctions
from toss.fitness.fitness_function_utils import _compute_squared_distance
from fitness_function_utils import estimate_covered_volume


def get_fitness_function(chosen_function: FitnessFunctions):
    """ 
    Returns user specified fitness function.
    """

    if chosen_function.value == 1:
        return target_altitude_distance
    
    elif chosen_function.value == 2:
        return close_distance_penalty
    
    elif chosen_function.value == 3:
        return far_distance_penalty
    
    elif chosen_function.value == 4:
        return covered_volume
    
    elif chosen_function.value == 5:
        return covered_volume_far_distance_penalty


def target_altitude_distance(args, positions: np.ndarray, timesteps: None) -> float:
    """ Computes average distance to target altitude for satellite positions along the trajectory.
    Args:
        args (dotmap.DotMap): Dotmap with information on radius of outer bounding sphere.
        positions (np.ndarray): (3,N) Array of positions along the trajectory.
        timesteps (None): (N) Array of time values for each position. 

    Returns:
        fitness (float): Average distance to target altitude.
    """    
    average_distance = np.mean(np.abs(_compute_squared_distance(positions, args.problem.target_squared_altitude)))
    fitness = 1/average_distance**2
    return fitness


def close_distance_penalty(args, positions: np.ndarray, timesteps: None) -> float:
    """ Computes average deviation from the inner-bounding sphere of satellite positions inside the inner bounding-sphere.
    Args:
        args (dotmap.DotMap): Dotmap with information on radius of outer bounding sphere.
        positions (np.ndarray): (3,N) Array of positions along the trajectory.
        timesteps (None): (N) Array of time values for each position.  

    Returns:
        fitness (float): Average distance to radius of inner bounding-sphere.
    """
    r = _compute_squared_distance(positions, args.problem.radius_inner_boundings_sphere)
    r = np.power(r[r<0], 2)
    fitness = 0
    for distance in r:
        fitness += 1/distance
    return fitness


def far_distance_penalty(args, positions: np.ndarray, timesteps: None) -> float:
    """ Computes average deviation from the outer-bounding sphere of satellite positions outside the outer bounding-sphere.
    Args:
        args (dotmap.DotMap): Dotmap with information on radius of outer bounding sphere.
        positions (np.ndarray): (3,N) Array of positions along the trajectory.
        timesteps (None): (N) Array of time values for each position. 
    Returns:
        fitness (float): Average distance to radius of outer bounding-sphere.
    """
    r = _compute_squared_distance(positions, args.problem.radius_inner_boundings_sphere)
    r = np.power(r[r>0], 2)
    fitness = 0
    for distance in r:
        fitness += 1/distance
    return fitness


def covered_volume(args, positions: np.ndarray, timesteps: None) -> float:
    """Computes the ration of unmeasured volume.
    Args:
        args (dotmap.DotMap): Dotmap with information on total measurable volume.
        positions (np.ndarray): (3,N) Array of positions along the trajectory.
        timesteps (None): (N) Array of time values for each position. 

    Returns:
        measured_volume_ratio (float): Ratio of unmeasured volume.
    """
    estimated_volume = estimate_covered_volume(positions)
    fitness = -(estimated_volume/args.problem.measurable_volume)
    return fitness


def covered_volume_far_distance_penalty(args, positions: np.ndarray, timesteps: np.ndarray) -> float:
    """ Returns combined fitness value of CoveredVolume and FarDistancePenalty
    Args:
        args (dotmap.DotMap): Dotmap with information on total measurable volume and radius of outer bounding sphere.
        positions (np.ndarray): (3,N) Array of positions along the trajectory.
        timesteps (None): (N) Array of time values for each position. 

    Returns:
        (float): Aggregate fitness value.
    """
    return (covered_volume(args,positions,timesteps) + far_distance_penalty(args,positions,timesteps))