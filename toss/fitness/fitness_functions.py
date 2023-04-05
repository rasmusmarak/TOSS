# Core packages
import numpy as np
import typing
from .fitness_function_enums import FitnessFunctions
from .fitness_function_utils import _compute_squared_distance, estimate_covered_volume

def get_fitness(chosen_fitness_function: FitnessFunctions, args, positions: np.ndarray, timesteps: None) -> float:
    """ Returns user specified fitness function.
    Args:
        chosen_function (FitnessFunctions): Chosen fitness function as defined in enum class FitnessFunctions.
        args (dotmap.DotMap): Dotmap with the required values for each defined fitness function.
        positions (np.ndarray): (3,N) Array of positions along the trajectory.
        timesteps (None): (N) Array of time values for each position. 

    Returns:
        (float): The evaluated fitness value correspondning to chosen_function.
    """

    if chosen_fitness_function == FitnessFunctions.TargetAltitudeDistance:
        return target_altitude_distance(args.problem.target_squared_altitude, positions)
    
    elif chosen_fitness_function == FitnessFunctions.CloseDistancePenalty:
        return close_distance_penalty(args.problem.radius_inner_bounding_sphere, positions)
    
    elif chosen_fitness_function == FitnessFunctions.FarDistancePenalty:
        return far_distance_penalty(args.problem.radius_outer_bounding_sphere, positions)
    
    elif chosen_fitness_function == FitnessFunctions.CoveredVolume:
        return covered_volume(args.problem.measurable_volume, positions)
    
    elif chosen_fitness_function == FitnessFunctions.CoveredVolumeFarDistancePenalty:
        return covered_volume_far_distance_penalty(args.problem.measurable_volume, args.problem.radius_outer_bounding_sphere, positions)


def target_altitude_distance(target_squared_altitude: float, positions: np.ndarray) -> float:
    """ Computes average distance to target altitude for satellite positions along the trajectory.
    Args:
        target_squared_altitude (float): Squared value of target altitude (from origin in cartesian frame)
        positions (np.ndarray): (3,N) Array of positions along the trajectory.

    Returns:
        fitness (float): Average distance to target altitude.
    """    
    average__squared_deviation_distance = np.mean(np.abs(_compute_squared_distance(positions, target_squared_altitude)))
    fitness = 1/average__squared_deviation_distance
    return fitness


def close_distance_penalty(radius_inner_bounding_sphere: float, positions: np.ndarray) -> float:
    """ Computes penalty depending on the position closest to the body and inside risk-zone.
    Args:
        radius_inner_bounding_sphere (float): Radius of inner bounding sphere.
        positions (np.ndarray): (3,N) Array of positions along the trajectory.

    Returns:
        Penalty (float): Penalty defined between [0,1].
    """
    # For each point along the trajectory, compute squared distance to inner sphere radius
    distance_squared = _compute_squared_distance(positions, radius_inner_bounding_sphere)
    
    # We only want to penalize positions that are inside inner-sphere (i.e risk-zone).
    # For positions inside sphere, identify the one farthest away from radius (i.e with greatest risk)
    maximum_distance = np.abs(np.min(distance_squared[distance_squared<0]))
    
    # Determine penalty P=[0,1] depending on distance. 
    penalty = maximum_distance/((maximum_distance + (radius_inner_bounding_sphere**2))/2)
    return penalty


def far_distance_penalty(radius_outer_bounding_sphere: float, positions: np.ndarray) -> float:
    """ Computes penalty depending on the position farthest away from measurement-zone.
    Args:
        radius_outer_bounding_sphere (float): Radius of outer bounding sphere.
        positions (np.ndarray): (3,N) Array of positions along the trajectory.

    Returns:
        Penalty (float): Penalty defined between [0,1].
    """
    # For each point along the trajectory, compute squared distance to inner sphere radius
    distance_squared = _compute_squared_distance(positions, radius_outer_bounding_sphere)
    
    # We only want to penalize positions that are outside outer-sphere (i.e measurement-zone).
    # For positions outside sphere, identify the one farthest away from radius (i.e least accurate measurment)
    maximum_distance = np.abs(np.max(distance_squared[distance_squared>0]))
    
    # Determine penalty P=[0,1] depending on distance. 
    penalty = maximum_distance/((maximum_distance + (radius_outer_bounding_sphere**2))/2)
    return penalty


def covered_volume(measurable_volume: float, positions: np.ndarray) -> float:
    """Computes the ration of unmeasured volume.
    Args:
        measurable_volume (float): Total measurable volume between two target altitudes defined by the radius of corresponding bounding spheres.
        positions (np.ndarray): (3,N) Array of positions along the trajectory.

    Returns:
        measured_volume_ratio (float): Ratio of unmeasured volume.
    """
    estimated_volume = estimate_covered_volume(positions)
    fitness = -(estimated_volume/measurable_volume)
    return fitness


def covered_volume_far_distance_penalty(measurable_volume: float, radius_outer_bounding_sphere: float, positions: np.ndarray) -> float:
    """ Returns combined fitness value of CoveredVolume and FarDistancePenalty
    Args:
        measurable_volume (float): Total measurable volume between two target altitudes defined by the radius of corresponding bounding spheres.
        radius_outer_bounding_sphere (float): Radius of outer bounding sphere.
        positions (np.ndarray): (3,N) Array of positions along the trajectory.

    Returns:
        (float): Aggregate fitness value.
    """
    return (covered_volume(measurable_volume,positions) + far_distance_penalty(radius_outer_bounding_sphere,positions))