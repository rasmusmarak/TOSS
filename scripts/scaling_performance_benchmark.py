import sys
sys.path.append('..')

# Core packages
from math import pi
import numpy as np
import time
import pygmo as pg


def load_udp(args, lower_bounds, upper_bounds, number_of_islands, population_size, number_of_generations):
    """
    Main function for optimizing the initial state for deterministic trajectories around a 
    small celestial body using a mesh.
    """
    # Load required modules
    from udp_initial_condition import udp_initial_condition

    start_time = time.time()

    # Setup User-Defined Problem (UDP)
    udp = udp_initial_condition(args, lower_bounds, upper_bounds)
    prob = pg.problem(udp)

    # Setup User-Defined Algorithm (UDA)
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
    #uda = pg.moead_gen(gen=number_of_generations)

    default_bfe = pg.bfe() # Default batch-fitness-evaluation for parallellized initialization of archipelago.
    uda.set_bfe(default_bfe)
    algo = pg.algorithm(uda)

    # Create archipelago 
    archi = pg.archipelago(n=number_of_islands, algo=algo, prob=prob, pop_size=population_size)
    archi.set_topology(pg.fully_connected(len(archi))) #scale number of vertices with number of islands for a fully-connected topology.
    archi.set_migration_type(pg.migration_type.broadcast)


    # Evolve archipelago (Optimization process)
    list_fitness_over_gen = np.empty((number_of_generations), dtype=np.float64)
    for i in range(0,number_of_generations):
            
        archi.evolve()
        print(archi)
        archi.wait()
        print(dir(archi))
        # Get champion
        f_champion_per_island = archi.get_champions_f()
        x_champion_per_island = archi.get_champions_x()
        f_champion_idx = np.where(f_champion_per_island == min(f_champion_per_island))[0]
        x_champion = x_champion_per_island[f_champion_idx[0]]
        f_champion = f_champion_per_island[f_champion_idx[0]][0]

        list_fitness_over_gen[i] = f_champion

    end_time = time.time()

    np.savetxt("fitness_over_gen.csv", list_fitness_over_gen, delimiter=",")

    # Compute elapsed time:
    elapsed_time = end_time - start_time

    # Get champion
    f_champion_per_island = archi.get_champions_f()
    x_champion_per_island = archi.get_champions_x()
    f_champion_idx = np.where(f_champion_per_island == min(f_champion_per_island))[0]
    x_champion = x_champion_per_island[f_champion_idx[0]]
    f_champion = f_champion_per_island[f_champion_idx[0]][0]

    return f_champion, x_champion, elapsed_time


def scaling(args, lower_bounds, upper_bounds, generations, islands, populations):
    """
    Generates results for a strong scaling test.
    """
    scaling_results = np.empty((8,len(islands)), dtype=np.float64)
    for i in range(0,len(islands)):
        number_of_islands = islands[i]
        population_size = populations[i]
        number_of_generations = generations[i]

        # Load udp
        f_champion, x_champion, elapsed_time = load_udp(args, lower_bounds, upper_bounds, number_of_islands, population_size, number_of_generations)
        
        # Store results
        scaling_results[0,i] = f_champion
        scaling_results[1:7,i] = x_champion
        scaling_results[7,i] = elapsed_time

        # For logs:
        print("Islands: ", number_of_islands, "   Pop: ", population_size,  "   Gen: ", number_of_generations)
        print("f: ", f_champion, "   elapsed time: ", elapsed_time)

    return scaling_results



def run_scaling_benchmark():
    # Load required modules
    from setup_parameters import setup_parameters

    # Load default parameters
    args, lower_bounds, upper_bounds = setup_parameters()

    # Adjust paramaters for benchmarking:
    args.problem.final_time = 10
    args.problem.number_of_maneuvers = 0
    args.problem.initial_time_step = 1


    # Initial test:
    generations = [10]
    islands = [1]
    populations = [70]
    initial_test_results = scaling(args, lower_bounds, upper_bounds, generations, islands, populations)
    np.savetxt("initial_test_results.csv", initial_test_results, delimiter=",")

    # Strong scaling, small run:
    generations = [10, 10, 10, 10, 10, 10]
    islands = [1, 2, 4, 8, 16, 32]
    populations = [320, 160, 80, 40, 20, 10]
    strong_scaling_results = scaling(args, lower_bounds, upper_bounds, generations, islands, populations)
    np.savetxt("strong_scaling_small_run.csv", strong_scaling_results, delimiter=",")

    # Strong scaling, large run:
    generations = [32, 32, 32, 32, 32, 32]
    islands = [1, 2, 4, 8, 16, 32]
    populations = [320, 160, 80, 40, 20, 10]
    strong_scaling_results = scaling(args, lower_bounds, upper_bounds, generations, islands, populations)
    np.savetxt("strong_scaling_large_run.csv", strong_scaling_results, delimiter=",")

    # Weak scaling
    generations = [32, 32, 32, 32, 32, 32]
    islands =  [1, 2, 4, 8, 16, 32]
    populations = [10, 10, 10, 10, 10, 10]
    weak_scaling_results = scaling(args, lower_bounds, upper_bounds, generations, islands, populations)
    np.savetxt("weak_scaling.csv", weak_scaling_results, delimiter=",")


if __name__ == "__main__":
    #run_scaling_benchmark()

    # Load required modules
    from setup_parameters import setup_parameters

    # Load default parameters
    args, lower_bounds, upper_bounds = setup_parameters()

    # Adjust paramaters for benchmarking:
    args.problem.final_time = 10
    args.problem.number_of_maneuvers = 0
    args.problem.initial_time_step = 1

    number_of_islands = 1#32
    population_size = 7#100
    number_of_generations = 4#50
    f_champion, x_champion, elapsed_time = load_udp(args, lower_bounds, upper_bounds, number_of_islands, population_size, number_of_generations)
    