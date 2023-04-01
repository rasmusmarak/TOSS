# Core packages
from math import pi
import numpy as np
import time
import pygmo as pg

# Load required modules
from toss.scripts.setup_parameters import setup_parameters
from toss.udp_initial_condition import udp_initial_condition

def load_udp(args, lower_bounds, upper_bounds, number_of_islands, population_size, number_of_generations):
    """
    Main function for optimizing the initial state for deterministic trajectories around a 
    small celestial body using a mesh.
    """

    start_time = time.time()

    # Setup User-Defined Problem (UDP)
    print("Setting up the UDP...")
    udp = udp_initial_condition(args, lower_bounds, upper_bounds)
    prob = pg.problem(udp)

    # Setup User-Defined Algorithm (UDA)
    print("Setting up UDA")
    uda = pg.gaco(
        number_of_generations, 
        args.algorithm.kernel_size, 
        args.algorithm.convergence_speed_parameter, 
        args.algorithm.oracle_parameter, 
        args.algorithm.accuracy_parameter, 
        args.algorithm.threshold_parameter, 
        args.algorithm.std_convergence_speed_parameter, 
        args.algorithm.improvement_stopping_criterion, 
        args.algorithm.evaluation_stopping_criterion, 
        args.algorithm.focus_parameter, 
        args.algorithm.memory_parameter)
    default_bfe = pg.bfe() # Default batch-fitness-evaluation for parallellized initialization of archipelago.
    uda.set_bfe(default_bfe)
    algo = pg.algorithm(uda)

    # Create archipelago 
    archi = pg.archipelago(n=number_of_islands, algo=algo, prob=prob, pop_size=population_size)
    archi.set_topology(pg.fully_connected(len(archi))) #scale number of vertices with number of islands for a fully-connected topology.
    archi.set_migration_type(pg.migration_type.broadcast)

    # Evolve archipelago (Optimization process)
    archi.evolve()
    print(archi)
    archi.wait()
    end_time = time.time()

    # Compute elapsed time:
    elapsed_time = end_time - start_time

    # Get champion
    f_champion_per_island = archi.get_champions_f()
    x_champion_per_island = archi.get_champions_x()
    f_champion_idx = np.where(f_champion_per_island == min(f_champion_per_island))[0]
    x_champion = x_champion_per_island[f_champion_idx[0]]
    f_champion = f_champion_per_island[f_champion_idx[0]][0]

    return f_champion, x_champion, elapsed_time



def strong_scaling(args, lower_bounds, upper_bounds, generations, islands, populations):
    strong_scaling_results = np.empty((8,len(islands)), dtype=np.float64)
    for i in range(0,len(islands)):
        number_of_islands = islands[i]
        population_size = populations[i]
        number_of_generations = generations[i]

        # Load udp
        f_champion, x_champion, elapsed_time = load_udp(args, lower_bounds, upper_bounds, number_of_islands, population_size, number_of_generations)
        
        # Store results
        strong_scaling_results[0,i] = f_champion
        strong_scaling_results[1:7,i] = x_champion
        strong_scaling_results[7,i] = elapsed_time

    return strong_scaling_results



def weak_scaling(args, lower_bounds, upper_bounds, generations, islands, populations):
    weak_scaling_results = np.empty((8,len(islands)), dtype=np.float64) # (8xlen(islands)) array with each column representing: [f, x, n_islands] (Here: f:1x1,  x:6x1, t:1x1)
    for i in range(0,len(islands)):
        number_of_islands = islands[i]
        population_size = populations[i]
        number_of_generations = generations[i]

        # Load udp
        f_champion, x_champion, elapsed_time = load_udp(args, lower_bounds, upper_bounds, number_of_islands, population_size, number_of_generations)
        
        # Store results
        weak_scaling_results[0,i] = f_champion
        weak_scaling_results[1:7,i] = x_champion
        weak_scaling_results[7,i] = elapsed_time

    return weak_scaling_results



def run_scaling_benchmark():

    # Load default parameters
    args, lower_bounds, upper_bounds = setup_parameters()

    # Adjust paramaters for benchmarking:
    args.problem.final_time = 1000
    args.problem.number_of_maneuvers = 0
    args.problem.initial_time_step = 1

    # Strong scaling, small run:
    generations = [10, 10, 10, 10, 10, 10]
    islands = [1, 2, 4, 8, 16, 32]
    populations = [320, 160, 80, 40, 20, 10]
    strong_scaling_results = strong_scaling(args, lower_bounds, upper_bounds, generations, islands, populations)
    np.savetxt("strong_scaling_small_run.csv", strong_scaling_results, delimiter=",")

    # Strong scaling, large run:
    generations = [32, 32, 32, 32, 32, 32]
    islands = [1, 2, 4, 8, 16, 32]
    populations = [320, 160, 80, 40, 20, 10]
    strong_scaling_results = strong_scaling(args, lower_bounds, upper_bounds, generations, islands, populations)
    np.savetxt("strong_scaling_large_run.csv", strong_scaling_results, delimiter=",")

    # Weak scaling
    generations = [32, 32, 32, 32, 32, 32]
    islands =  [1, 2, 4, 8, 16, 32]
    populations = [10, 10, 10, 10, 10, 10]
    weak_scaling_results = weak_scaling(args, lower_bounds, upper_bounds, generations, islands, populations)
    np.savetxt("weak_scaling.csv", weak_scaling_results, delimiter=",")


if __name__ == "__main__":
    run_scaling_benchmark()