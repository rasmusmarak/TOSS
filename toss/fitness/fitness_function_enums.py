# General
from enum import Enum
 
class FitnessFunctions(Enum):
    """
        Defines options of different types of fitness values to be minimized in the objective function.

        1. TargetAltitudeDistance:   
            Computes the average squared distance from the satellites position 
            on the trajetory to the squared value of some given target altitude.

        2. CloseDistancePenalty:   
            Computes the average squared distance to the radius of the inner 
            bounding-sphere for satellite positions inside the risk-zone.

        3. FarDistancePenalty:
            Computes the average squared distance to the radius of the outer 
            bounding-sphere for satellite positions outside the measurement-zone.
        
        4. CoveredVolume:
            Computes the ratio of unmeasured volume iniside the search space
            defined by the radius of some inner and outer bounding sphere.
    
        5. CoveredVolumeFarDistancePenalty:
            Combination of (3) and (4)
    """
    TargetAltitudeDistance = 1
    CloseDistancePenalty = 2
    FarDistancePenalty = 3
    CoveredVolume = 4
    CoveredVolumeFarDistancePenalty = 5
