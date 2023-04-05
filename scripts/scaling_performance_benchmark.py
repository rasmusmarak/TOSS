import sys
sys.path.append("..")
sys.path.append("../..")

# Core packages
from math import pi
import numpy as np
import time
import pygmo as pg

# Load required modules
from udp_initial_condition import udp_initial_condition
from toss.setup_parameters import setup_parameters

def load_udp(args, lower_bounds, upper_bounds, number_of_islands, population_size, number_of_generations):
    """
    Main function for optimizing the initial state for deterministic trajectories around a 
    small celestial body using a mesh.
    """
    # Setup User-Defined Problem (UDP)
    udp = udp_initial_condition(args, lower_bounds, upper_bounds)
    begin_time = time.time()
    prob = pg.problem(udp)
    stop_time = time.time()

    print("serialization ", stop_time-begin_time)

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

    multi_process_bfe = pg.mp_bfe()
    multi_process_bfe.resize_pool(number_of_islands)
    bfe = pg.bfe(multi_process_bfe) 
    uda.set_bfe(bfe)
    algo = pg.algorithm(uda)

    # Create archipelago 
    start_time = time.time()
    pop = pg.population(prob, size=population_size)
    pop_time = time.time()
    
    # Evolve (Optimization process)
    algo.set_verbosity(1)
    pop = algo.evolve(pop)
    end_time = time.time()

    # Logs for output
    f_champion = pop.champion_f
    x_champion = pop.champion_x
    print("Champion fitness value: ", pop.champion_f) 
    print("Champion chromosome: ", pop.champion_x) 
    
    # Shutdown pool to avoid mp_bfe bug for python==3.8
    multi_process_bfe.shutdown_pool()

    # Compute elapsed times:
    elapsed_time = end_time - start_time
    population_time = pop_time - start_time
    evolve_time = end_time - pop_time

    print("Elapsed time: ", elapsed_time)
    print("Population time: ", population_time)
    print("Evolve time: ", evolve_time)

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
    populations = [640, 320, 160, 80, 40, 20]
    strong_scaling_results = scaling(args, lower_bounds, upper_bounds, generations, islands, populations)
    np.savetxt("strong_scaling_small_run.csv", strong_scaling_results, delimiter=",")

    # Strong scaling, large run:
    generations = [40, 40, 40, 40, 40, 40]
    islands = [1, 2, 4, 8, 16, 32]
    populations = [640, 320, 160, 80, 40, 20]
    strong_scaling_results = scaling(args, lower_bounds, upper_bounds, generations, islands, populations)
    np.savetxt("strong_scaling_large_run.csv", strong_scaling_results, delimiter=",")

    # Weak scaling
    generations = [40, 40, 40, 40, 40, 40]
    islands =  [1, 2, 4, 8, 16, 32]
    populations = [20, 20, 20, 20, 20, 20]
    weak_scaling_results = scaling(args, lower_bounds, upper_bounds, generations, islands, populations)
    np.savetxt("weak_scaling.csv", weak_scaling_results, delimiter=",")


if __name__ == "__main__":
    run_scaling_benchmark()    