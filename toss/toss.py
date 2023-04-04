import sys
sys.path.append("..")
sys.path.append("../..")

# Core packages
from math import pi
import numpy as np
import pygmo as pg

# For cProfile evaluation
import cProfile
import pstats

# Load required modules
from udp_initial_condition import udp_initial_condition
from scripts.setup_parameters import setup_parameters



def load_udp(args, lower_bounds, upper_bounds):
    """
    Main function for optimizing the initial state for deterministic trajectories around a 
    small celestial body using a mesh.
    """

    # Setup User-Defined Problem (UDP)
    print("Setting up the UDP...")
    udp = udp_initial_condition(args, lower_bounds, upper_bounds)
    prob = pg.problem(udp)

    # Setup optimization algorithm
    print("Setting up the optimization algorithm...")
    assert args.optimization.population_size >= 7

    # Setup User-Defined Algorithm (UDA)
    print("Setting up UDA")
    uda = pg.gaco(
        args.optimization.number_of_generations, 
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
    multi_process_bfe.resize_pool(args.optimization.number_of_islands)
    bfe = pg.bfe(multi_process_bfe) 
    uda.set_bfe(bfe)
    algo = pg.algorithm(uda)
    
    # Setup population
    pop = pg.population(prob, size=args.optimization.population_size)

    # Evolve archipelago (Optimization process)
    algo.set_verbosity(1)
    pop = algo.evolve(pop)

    # Logs for output
    champion_f = pop.champion_f
    champion_x = pop.champion_x
    print("Champion fitness value: ", champion_f) 
    print("Champion chromosome: ", champion_x) 

    # Shutdown pool to avoid mp_bfe bug for python==3.8
    multi_process_bfe.shutdown_pool()

    return champion_f, champion_x



def main():
    args, lower_bounds, upper_bounds = setup_parameters()
    champion_f, champion_x = load_udp(args, lower_bounds, upper_bounds)


if __name__ == "__main__":
    cProfile.run("main()", "output.dat")

    with open("output_time.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("time").print_stats()
    
    with open("output_calls.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("calls").print_stats()
        