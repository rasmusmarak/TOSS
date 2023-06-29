
# Core packages
from math import pi
import numpy as np
import time
import pygmo as pg

# Load required modules
from toss.optimization.udp_initial_condition import udp_initial_condition
from toss.optimization.setup_parameters import setup_parameters
from toss.optimization.setup_state import setup_initial_state_domain


def evolve_fitness_over_gen(args, initial_condition, lower_bounds, upper_bounds):
    """
    Main function for optimizing the initial state for deterministic trajectories around a 
    small celestial body using a mesh.
    """
    # Setup User-Defined Problem (UDP)
    print("Setting up the UDP...")
    udp = udp_initial_condition(args, initial_condition, lower_bounds, upper_bounds)
    prob = pg.problem(udp)

    # Setup optimization algorithm
    print("Setting up the optimization algorithm...")
    assert args.optimization.population_size >= 7

    # Setup timer
    time_start = time.time()

    # Setup User-Defined Algorithm (UDA)
    print("Setting up UDA")
    uda = pg.gaco(
        1, 
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

    # Setup BFE machinery
    multi_process_bfe = pg.mp_bfe()
    multi_process_bfe.resize_pool(args.optimization.number_of_threads)
    bfe = pg.bfe(multi_process_bfe) 
    uda.set_bfe(bfe)
    algo = pg.algorithm(uda)
    
    # Setup population
    pop = pg.population(prob, size=args.optimization.population_size)
    
    # Evolve (Optimization process)
    fitness_list = []
    for i in range(args.optimization.number_of_generations):
        pop = algo.evolve(pop)
        fitness_list.append(pop.get_f()[pop.best_idx()])
        print("Generations: ", i+1, " "*10, "Best: ", fitness_list[len(fitness_list)-1])
    end_time = time.time()
 
    # Logs for output
    print("Champion fitness value: ", pop.champion_f) 
    print("Champion chromosome: ", pop.champion_x) 
    
    # Shutdown pool to avoid mp_bfe bug for python==3.8
    multi_process_bfe.shutdown_pool()

    # Compute elapsed times:
    evolve_time = end_time - time_start
    print("Evolve time: ", evolve_time)
    
    return fitness_list


def fitness_over_generations():
    
    # Setup problem parameters (as DotMaP)
    args = setup_parameters()
    args.problem.final_time = 1000

    args.problem.number_of_maneuvers = 0
    args.problem.number_of_spacecrafts = 1

    args.problem.measurement_period = 100
    args.problem.max_velocity_scaling_factor = 40
    args.problem.penalty_scaling_factor = 0.1

    args.optimization.number_of_generations = 1000
    args.optimization.population_size = 120
    args.optimization.number_of_threads = 40
    
    # Setup initial state space
    initial_condition = []
    lower_bounds, upper_bounds = setup_initial_state_domain(initial_condition, 
                                                            args.problem.start_time, 
                                                            args.problem.final_time, 
                                                            args.problem.number_of_maneuvers, 
                                                            args.problem.number_of_spacecrafts)

    

    fitness_list = evolve_fitness_over_gen(args, initial_condition, lower_bounds, upper_bounds)
    return fitness_list


if __name__ == "__main__":
    fitness_list = fitness_over_generations()
    np.savetxt("fitness_over_generations.csv", np.array(fitness_list), delimiter=",")