# Core packages
import numpy as np

def get_fitness_function(chosen_function):
    """ 
    Returns user specified fitness function.
    """

    if chosen_function.value == 1:
        return TargetAltitudeDistance
    
    elif chosen_function.value == 2:
        return CloseDistancePenalty
    
    elif chosen_function.value == 3:
        return FarDistancePenalty
    
    elif chosen_function.value == 4:
        return CoveredVolume
    
    elif chosen_function.value == 5:
        return CoveredVolumeFarDistancePenalty


def TargetAltitudeDistance(args, positions: np.ndarray, timesteps: None) -> float:
    """ Computes average distance to target altitude for satellite positions along the trajectory.
    Args:
        args (dotmap.DotMap): Dotmap with information on target squared altitude of spacecraft.
        positions (np.ndarray): A set of positions along the trajectory.
        timesteps (None): Time values for each position. 

    Returns:
        fitness (float): Average distance to target altitude.
    """    
    fitness = np.mean(np.abs(positions[0,:]**2 + positions[1,:]**2 + positions[2,:]**2 -args.problem.target_squared_altitude))
    return fitness


def CloseDistancePenalty(args, positions: np.ndarray, timesteps: None) -> float:
    """ Computes average deviation from the inner-bounding sphere of satellite positions inside the inner bounding-sphere.
    Args:
        args (dotmap.DotMap): Dotmap with information on radius of inner bounding sphere.
        positions (np.ndarray): A set of positions along the trajectory.
        timesteps (None): Time values for each position. 

    Returns:
        fitness (float): Average distance to radius of inner bounding-sphere.
    """
    r = positions[0:3, :]
    r = (r[0,:]**2 + r[1,:]**2 + r[2, :]**2) - args.problem.radius_inner_boundings_sphere
    r[r>0] = 0 #removes positions outside the risk-zone.
    fitness = np.mean(r)
    return fitness


def FarDistancePenalty(args, positions: np.ndarray, timesteps: None) -> float:
    """ Computes average deviation from the outer-bounding sphere of satellite positions outside the outer bounding-sphere.
    Args:
        args (dotmap.DotMap): Dotmap with information on radius of outer bounding sphere.
        positions (np.ndarray): A set of positions along the trajectory.
        timesteps (None): Time values for each position. 
    Returns:
        fitness (float): Average distance to radius of outer bounding-sphere.
    """
    r = positions[0:3, :]
    r = (r[0,:]**2 + r[1,:]**2 + r[2, :]**2) - args.problem.radius_outer_boundings_sphere
    r[r<0] = 0 #removes all positions inside the measurement-zone.
    fitness = np.mean(r)
    return fitness


def CoveredVolume(args, positions: np.ndarray, timesteps: None) -> float:
    """Computes the ration of unmeasured volume.
    Args:
        args (dotmap.DotMap): Dotmap with information on total measurable volume.
        positions (np.ndarray): A set of positions along the trajectory.
        timesteps (None): Time values for each position. 

    Returns:
        measured_volume_ratio (float): Ratio of unmeasured volume.
    """
    from fitness_function_utils import estimate_covered_volume
    estimated_volume = estimate_covered_volume(positions)
    fitness = -(estimated_volume/args.problem.measurable_volume)
    return fitness


def CoveredVolumeFarDistancePenalty(args, positions: np.ndarray, timesteps: np.ndarray) -> float:
    """ Returns combined fitness value of CoveredVolume and FarDistancePenalty
    Args:
        args (dotmap.DotMap): Dotmap with information on total measurable volume and radius of outer bounding sphere.
        positions (np.ndarray): A set of positions along the trajectory.
        timesteps (None): Time values for each position. 

    Returns:
        (float): Aggregate fitness value.
    """
    return (CoveredVolume(args,positions,timesteps) + FarDistancePenalty(args,positions,timesteps))