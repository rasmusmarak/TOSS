# Core packages
import numpy as np
from typing import Callable
from toss.fitness.fitness_function_enums import FitnessFunctions
from toss.fitness.fitness_function_utils import _compute_squared_distance
from fitness_function_utils import estimate_covered_volume


def get_fitness_function(chosen_function: FitnessFunctions) -> Callable:
    """ Returns user specified fitness function.
    Args:
        chosen_function (FitnessFunctions): Chosen fitness function defined in enum class FitnessFunctions.

    Returns:
        (callable): The correct fitness function correspondning to chosen_function.
    """

    if chosen_function == FitnessFunctions.TargetAltitudeDistance:
        return target_altitude_distance
    
    elif chosen_function.value == FitnessFunctions.CloseDistancePenalty:
        return close_distance_penalty
    
    elif chosen_function.value == FitnessFunctions.FarDistancePenalty:
        return far_distance_penalty
    
    elif chosen_function.value == FitnessFunctions.CoveredVolume:
        return covered_volume
    
    elif chosen_function.value == FitnessFunctions.CoveredVolumeFarDistancePenalty:
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
    """ Computes penalty depending on the position closest to the body and inside risk-zone.
    Args:
        args (dotmap.DotMap): Dotmap with information on radius of outer bounding sphere.
        positions (np.ndarray): (3,N) Array of positions along the trajectory.
        timesteps (None): (N) Array of time values for each position.  

    Returns:
        Penalty (float): Penalty defined between [0,1].
    """
    # For each point along the trajectory, compute squared distance to inner sphere radius
    distance_squared = _compute_squared_distance(positions, args.problem.radius_inner_boundings_sphere)
    
    # We only want to penalize positions that are inside inner-sphere (i.e risk-zone).
    # For positions inside sphere, identify the one farthest away from radius (i.e with greatest risk)
    maximum_distance = np.abs(np.min(distance_squared[distance_squared<0]))
    
    # Determine penalty P=[0,1] depending on distance. 
    penalty = maximum_distance/((maximum_distance + args.problem.radius_inner_boundings_sphere)/2)
    return penalty


def far_distance_penalty(args, positions: np.ndarray, timesteps: None) -> float:
    """ Computes penalty depending on the position farthest away from measurement-zone.
    Args:
        args (dotmap.DotMap): Dotmap with information on radius of outer bounding sphere.
        positions (np.ndarray): (3,N) Array of positions along the trajectory.
        timesteps (None): (N) Array of time values for each position. 
    Returns:
        Penalty (float): Penalty defined between [0,1].
    """
    # For each point along the trajectory, compute squared distance to inner sphere radius
    distance_squared = _compute_squared_distance(positions, args.problem.radius_outer_boundings_sphere)
    
    # We only want to penalize positions that are outside outer-sphere (i.e measurement-zone).
    # For positions outside sphere, identify the one farthest away from radius (i.e least accurate measurment)
    maximum_distance = np.abs(np.max(distance_squared[distance_squared>0]))
    
    # Determine penalty P=[0,1] depending on distance. 
    penalty = maximum_distance/((maximum_distance + args.problem.radius_inner_boundings_sphere)/2)
    return penalty


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