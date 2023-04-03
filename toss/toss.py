import sys
sys.path.append('..')

# Core packages
from math import pi
import numpy as np
import pygmo as pg

# For cProfile evaluation
import cProfile
import pstats




def load_udp(args, lower_bounds, upper_bounds):
    """
    Main function for optimizing the initial state for deterministic trajectories around a 
    small celestial body using a mesh.
    """
    # Load required modules
    from udp_initial_condition import udp_initial_condition

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
    default_bfe = pg.bfe() # Default batch-fitness-evaluation for parallellized initialization of archipelago.
    uda.set_bfe(default_bfe)
    algo = pg.algorithm(uda)

    # Create archipelago 
    archi = pg.archipelago(n=args.optimization.number_of_islands, algo=algo, prob=prob, pop_size=args.optimization.population_size)
    archi.set_topology(pg.fully_connected(len(archi))) #scale number of vertices with number of islands for a fully-connected topology.
    archi.set_migration_type(pg.migration_type.broadcast)

    # Evolve archipelago (Optimization process)
    archi.evolve()
    print(archi)
    archi.wait()

    # Get champion
    f_champion_per_island = archi.get_champions_f()
    x_champion_per_island = archi.get_champions_x()
    print("Champion fitness value: ", f_champion_per_island)
    print("Champion chromosome: ", x_champion_per_island)

    f_champion_idx = np.where(f_champion_per_island == min(f_champion_per_island))[0]
    x_champion = x_champion_per_island[f_champion_idx[0]]
    f_champion = f_champion_per_island[f_champion_idx[0]][0]

    return x_champion






def main():
    # Load required modules
    from scripts.setup_parameters import setup_parameters

    args, lower_bounds, upper_bounds = setup_parameters()
    load_udp(args, lower_bounds, upper_bounds)


if __name__ == "__main__":
    cProfile.run("main()", "output.dat")

    with open("output_time.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("time").print_stats()
    
    with open("output_calls.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("calls").print_stats()
        