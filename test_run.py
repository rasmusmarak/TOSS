# Module scripts
from toss import compute_trajectory
from toss import compute_motion, setup_spin_axis, rotate_point
from toss import get_trajectory_fixed_step
from toss import FitnessFunctions
from toss import get_fitness
from toss import create_mesh
from toss import setup_parameters
from toss import plot_UDP_3D, plot_UDP_2D, fitness_over_generations, fitness_over_time, distance_deviation_over_time
from toss import compute_space_coverage, create_spherical_tensor_grid

# Core packages
from dotmap import DotMap
import numpy as np
from math import pi
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D


args = setup_parameters()

# Concatenate initial position with chromosomes
x = np.genfromtxt('test_champion_x.csv', delimiter=',')
initial_conditions = np.array_split([-135.13402075, -4089.53592604, 6050.17636635]*args.problem.number_of_spacecrafts, args.problem.number_of_spacecrafts)

list_of_spacecrafts = []
for i in range(0, len(x)):
    list_of_spacecrafts.append(np.concatenate((initial_conditions[i], x[i])))


# Resample and store trajectory for each spacecraft with a fixed time-step delta t
positions = None
velocities = None
timesteps = None
maneuver_times = None
maneuver_vectors = None

for counter, spacecraft in enumerate(list_of_spacecrafts):

    # Compute trajectory
    collision_detected, list_of_ode_objects, _ = compute_trajectory(spacecraft, args, compute_motion)
    
    # Resample trajectory for a fixed time-step delta t
    spacecraft_positions, spacecraft_velocities, spacecraft_timesteps = get_trajectory_fixed_step(args, list_of_ode_objects)

    # Manage maneuvers
    maneuvers = np.array_split(spacecraft[7:],args.problem.number_of_maneuvers)
    maneuver_t = np.zeros((len(maneuvers)))
    maneuver_v = np.zeros((3,len(maneuvers)))
    maneuver_p = np.zeros((3,len(maneuvers)))
    maneuver_unit_v = np.zeros((3,len(maneuvers)))
    for idx, maneuver in enumerate(maneuvers):
        # store time of maneuver and corresponding control vector
        maneuver_t[idx] = maneuver[0]
        maneuver_v[:,idx] = maneuver[1]*maneuver[2:]

        # Find position where maneuver was engaged as well as the maneuver unit vector
        maneuver_time_idx = (np.abs(np.asarray(spacecraft_timesteps) - maneuver[0])).argmin()
        maneuver_p[:,idx] = spacecraft_positions[:,maneuver_time_idx]
        maneuver_unit_v[:,idx] = maneuver_v[:,idx] / np.linalg.norm(maneuver_v[:,idx])


    # Store information
    if counter == 0:
        positions = spacecraft_positions
        velocities = spacecraft_velocities
        timesteps = spacecraft_timesteps
        maneuver_times = maneuver_t
        maneuver_vectors = maneuver_v
        maneuver_positions = maneuver_p
        maneuver_unit_vectors = maneuver_unit_v

    else:
        positions = np.hstack((positions, spacecraft_positions))
        velocities = np.hstack((velocities, spacecraft_velocities))
        maneuver_times = np.hstack((maneuver_times, maneuver_t))
        maneuver_vectors = np.hstack((maneuver_vectors, maneuver_v))
        maneuver_positions = np.hstack((maneuver_positions, maneuver_p))
        maneuver_unit_vectors = np.hstack((maneuver_unit_vectors, maneuver_unit_v))



# Compute fitness
fitness = get_fitness(FitnessFunctions.CoveredSpaceCloseDistancePenaltyFarDistancePenalty, args, positions, velocities, timesteps)
print("CoveredSpaceCloseDistancePenaltyFarDistancePenalty: ", fitness)

clospenalty = get_fitness(FitnessFunctions.CloseDistancePenalty, args, positions, velocities, timesteps)
print("CloseDistancePenalty: ", clospenalty)

farpenalty = get_fitness(FitnessFunctions.FarDistancePenalty, args, positions, velocities, timesteps)
print("FarDistancePenalty: ", farpenalty)

coverage = get_fitness(FitnessFunctions.CoveredSpace, args, positions, velocities, timesteps)
print("CoveredSpace: ", coverage)