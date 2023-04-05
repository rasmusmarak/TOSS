import sys
sys.path.append("..")
sys.path.append("../..")

# Core packages
from math import pi
import numpy as np
import time
import pygmo as pg

# Load required modules
from toss.udp_initial_condition import udp_initial_condition
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


if __name__ == "__main__":

    ########## Run initial test ##########
    # Load default parameters
    args, lower_bounds, upper_bounds = setup_parameters()

    # Adjust paramaters for benchmarking:
    args.problem.final_time = 10
    args.problem.number_of_maneuvers = 0
    args.problem.initial_time_step = 1

    number_of_islands = 4
    population_size = 20
    number_of_generations = 20
    load_udp(args, lower_bounds, upper_bounds, number_of_islands, population_size, number_of_generations)
