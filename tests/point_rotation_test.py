
""" This test checks whether or not a point rotation is performed correctly """

# Import required modules
from toss import setup_spin_axis, rotate_point

# Core packages
from dotmap import DotMap
import math
from math import pi
import numpy as np
from numpy.random import randint, rand


def test_rotation_of_point():
    """
    This is a test verify that the chosen method, rotate_point, as defined
    in EquationsOfMotion rotates a given point and axis correctly. To verify 
    the results of rotate_point, the method is compared with the analytical 
    rotation given by the Euler-Rodrigues formula. This is done for a set of
    random position vectors and rotation times. The number of such comparisons
    is provided by the user in terms of n_max.
    
    Sources:
        - https://en.wikipedia.org/wiki/Euler–Rodrigues_formula
        - Jian S. Dai, Euler–Rodrigues formula variations, quaternion conjugation 
        and intrinsic connections, Mechanism and Machine Theory, Volume 92, 2015, 
        Pages 144-152, ISSN 0094-114X, https://doi.org/10.1016/j.mechmachtheory.2015.03.004.

    """

    # Body parameters
    args = DotMap(
        body=DotMap(_dynamic=False),
        _dynamic=False)
    args.body.density = 533                  # https://sci.esa.int/web/rosetta/-/14615-comet-67p
    args.body.mu = 665.666                   # Gravitational parameter for 67P/C-G
    args.body.declination = 64               # [degrees] https://sci.esa.int/web/rosetta/-/14615-comet-67p
    args.body.right_ascension = 69           # [degrees] https://sci.esa.int/web/rosetta/-/14615-comet-67p
    args.body.spin_period = 12.06*3600       # [seconds] https://sci.esa.int/web/rosetta/-/14615-comet-67p
    args.body.spin_velocity = (2*pi)/args.body.spin_period
    args.body.spin_axis = setup_spin_axis(args)

    # Define number of rotations to evaluate
    n_max = 10

    for i in range(0, n_max-1):
        
        # Generate random 3-dim array representing a position [m]
        x = randint(0, 20000, 3)
        
        # Generate a random time [s] to define the rotation angle.
        t = rand(1)
        t = t[0]

        ######### Rotation using Quaternion #########
        # Rotation of point: 
        rotated_position_quaternion = rotate_point(t, x, args)

        #########    Analtical rotation    #########
        # Define analytical rotation (euler-rodrigues):
        axis = np.asarray(args.body.spin_axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos((2*pi - (args.body.spin_velocity*t))/2.0)
        b, c, d = -axis * math.sin((2*pi - (args.body.spin_velocity*t))/ 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

        rotation_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

        rotated_position_analytical = np.dot(rotation_matrix, x)

        # Check if both methods give equal rotation
        assert all(np.isclose(rotated_position_analytical,rotated_position_quaternion,rtol=1e-5, atol=1e-5))