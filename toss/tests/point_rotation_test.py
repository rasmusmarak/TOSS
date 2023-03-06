
""" This test checks whether or not a point rotation is performed correctly """

import sys

sys.path.append("../..")
import toss
from toss import mesh_utility
from toss.EquationsOfMotion import EquationsOfMotion

# Core packages
from dotmap import DotMap
import math
from math import pi, radians
import numpy as np
from numpy.random import rand

# For computing rotations of orbits
from pyquaternion import Quaternion

def test_rotation_of_point():
    """
    This is a test verify that the chosen method, rotate_point, as defined
    in EquationsOfMotion rotates a given point and axis correctly. To verify 
    the results of rotate_point, the method is compared with the analytical 
    rotation given by the Euler-Rodrigues formula.
    
    Sources:
        - https://en.wikipedia.org/wiki/Euler–Rodrigues_formula
        - Jian S. Dai, Euler–Rodrigues formula variations, quaternion conjugation 
        and intrinsic connections, Mechanism and Machine Theory, Volume 92, 2015, 
        Pages 144-152, ISSN 0094-114X, https://doi.org/10.1016/j.mechmachtheory.2015.03.004.

    """

    # Body parameters
    body_args = DotMap()
    body_args.density = 533                  # https://sci.esa.int/web/rosetta/-/14615-comet-67p
    body_args.mu = 665.666                   # Gravitational parameter for 67P/C-G
    body_args.declination = 64               # [degrees] https://sci.esa.int/web/rosetta/-/14615-comet-67p
    body_args.right_ascension = 69           # [degrees] https://sci.esa.int/web/rosetta/-/14615-comet-67p
    body_args.spin_period = 12.06*3600       # [seconds] https://sci.esa.int/web/rosetta/-/14615-comet-67p

    # Creating the mesh (TetGen)
    _, mesh_vertices, mesh_faces, _ = mesh_utility.create_mesh()
    # Setup equations of motion class
    eq_of_motion = EquationsOfMotion(mesh_vertices, mesh_faces, body_args)


    # Define number of rotations to evaluate
    n_max = 10
    for i in range(0, n_max-1):
        
        # Generate random 3-dim array representing a position [m]
        x = rand(0, 20000, 3)
        
        # Generate a random time [s] to define the rotation angle.
        t = rand(1)
        t = t[0]

        ######### Rotation using Quaternion #########
        # Rotation of point: 
        rotated_position_quaternion = eq_of_motion.rotate_point(t, x)
        #############################################

        #########    Analtical rotation    #########
        # Define spin axis as in EquationsOfMotion:
        spin_velocity = (2*pi)/body_args.spin_period
        q_dec = Quaternion(axis=[1,0,0], angle=radians(body_args.declination)) # Rotate spin axis according to declination
        q_ra = Quaternion(axis=[0,0,1], angle=radians(body_args.right_ascension)) # Rotate spin axis accordining to right ascension
        q_axis = q_dec * q_ra  # Composite rotation of q1 then q2 expressed as standard multiplication
        spin_axis = q_axis.rotate([0,0,1])


        # Define analytical rotation (euler-rodrigues):
        axis = np.asarray(spin_axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos((2*pi - (spin_velocity*t))/2.0)
        b, c, d = -axis * math.sin((2*pi - (spin_velocity*t))/ 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

        rotation_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

        rotated_position_analytical = np.dot(rotation_matrix, x)

        # Check if both methods give equal rotation
        assert all(np.isclose(rotated_position_analytical,rotated_position_quaternion,rtol=1e-5, atol=1e-5))
