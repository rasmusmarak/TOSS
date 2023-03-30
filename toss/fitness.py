# General
from enum import Enum
 
class FitnessScheme(Enum):
        """
         Defines options of different types of fitness values to be minimized in the objective function.

            1. distance_to_target_altitude:   
                Computes the average squared distance from the satellites position 
                on the trajetory to the squared value of some given target altitude.

            2. inner_sphere_entries:   
                Computes the average squared distance to the radius of the inner 
                bounding-sphere for satellite positions inside the risk-zone.

            3. outer_sphere_exits:
                Computes the average squared distance to the radius of the outer 
                bounding-sphere for satellite positions outside the measurement-zone.
            
            4. unmeasured_volume:
                Computes the ratio of unmeasured volume iniside the search space
                defined by the radius of some inner and outer bounding sphere.
        """
        distance_to_target_altitude = 1
        inner_sphere_entries = 2
        outer_sphere_exits = 3
        unmeasured_volume = 4