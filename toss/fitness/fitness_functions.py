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
        return covered_volume(args.problem.maximal_measurement_sphere_volume, positions)

    elif chosen_fitness_function == FitnessFunctions.TotalCoveredVolume:
        return total_covered_volume(args.problem.total_measurable_volume, positions)

    elif chosen_fitness_function == FitnessFunctions.CoveredVolumeFarDistancePenalty:
        return covered_volume_far_distance_penalty(args.problem.maximal_measurement_sphere_volume, args.problem.radius_outer_bounding_sphere, positions)
    
    elif chosen_fitness_function == FitnessFunctions.CoveredVolumeCloseDistancePenaltyFarDistancePenalty:
        return covered_volume_close_distance_penalty_far_distance_penalty(args.problem.maximal_measurement_sphere_volume, args.problem.radius_inner_bounding_sphere, args.problem.radius_outer_bounding_sphere, positions)


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
    distance_squared = distance_squared[distance_squared<0]
    if len(distance_squared) == 0:
        return 0
    else:
        # For positions inside sphere, identify the one farthest away from radius (i.e with greatest risk)
        maximum_distance = np.abs(np.min(distance_squared))
        
        # Determine penalty P=[0,1] depending on distance. 
        penalty = (maximum_distance/(radius_inner_bounding_sphere**2))**(1/4)

        return penalty


def far_distance_penalty(radius_outer_bounding_sphere: float, positions: np.ndarray) -> float:
    """ Computes penalty depending on the mean distance to the measurement-zone.
    Args:
        radius_outer_bounding_sphere (float): Radius of outer bounding sphere.
        positions (np.ndarray): (3,N) Array of positions along the trajectory.

    Returns:
        Penalty (float): Penalty defined between [0,1].
    """
    # For each point along the trajectory, compute squared distance to inner sphere radius
    distance_squared = _compute_squared_distance(positions, radius_outer_bounding_sphere)
    
    # We only want to penalize positions that are outside outer-sphere (i.e measurement-zone).
    distance_squared = distance_squared[distance_squared>0]
    if len(distance_squared)==0:
        return 0
    else:
        # For positions outside sphere, identify the mean distande from radius
        mean_distance = np.mean(distance_squared)
        
        # Determine penalty depending on mean distance. 
        penalty = (mean_distance/(radius_outer_bounding_sphere**2))**(1/4)
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

def covered_volume_far_distance_penalty(maximal_measurement_sphere_volume: float, radius_outer_bounding_sphere: float, positions: np.ndarray) -> float:
    """ Returns combined fitness value of CoveredVolume and FarDistancePenalty
    Args:
        maximal_measurement_sphere_volume (float): Total measurable volume for a measurement sphere at the inner bounding sphere radius positioned at estimated greatest gravitational influence from the body.
        radius_outer_bounding_sphere (float): Radius of outer bounding sphere.
        positions (np.ndarray): (3,N) Array of positions along the trajectory.

    Returns:
        (float): Aggregate fitness value.
    """
    return (covered_volume(maximal_measurement_sphere_volume,positions) + far_distance_penalty(radius_outer_bounding_sphere,positions))


def covered_volume_close_distance_penalty_far_distance_penalty(maximal_measurement_sphere_volume: float, radius_inner_bounding_sphere: float, radius_outer_bounding_sphere: float, positions: np.ndarray):
    """ Returns aggregate fitness of covered_volume, close_distance_penalty and far_distance_penalty.
    Args:
        maximal_measurement_sphere_volume (float): Total measurable volume for a measurement sphere at the inner bounding sphere radius positioned at estimated greatest gravitational influence from the body.
        radius_inner_bounding_sphere (float): Radius of inner bounding sphere.
        radius_outer_bounding_sphere (float): Radius of outer bounding sphere.
        positions (np.ndarray): (3,N) Array of positions along the trajectory.

    Returns:
        (float): Aggregate fitness value.
    """
    return (covered_volume(maximal_measurement_sphere_volume,positions) + close_distance_penalty(radius_inner_bounding_sphere, positions) + far_distance_penalty(radius_outer_bounding_sphere,positions))