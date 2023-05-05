# Core packages
import pygmo as pg
import cProfile
import pstats
import time
import numpy as np

# Load required modules
from toss.optimization.udp_initial_condition import udp_initial_condition
from toss.optimization.setup_parameters import setup_parameters
from toss.optimization.setup_state import setup_initial_state_domain

def load_udp(args, initial_condition, lower_bounds, upper_bounds):
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
    run_time_start = time.time()

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
    multi_process_bfe = pg.mp_bfe() #pg.ipyparallel_bfe() #n=args.optimization.number_of_threads
    multi_process_bfe.resize_pool(args.optimization.number_of_threads)
    bfe = pg.bfe(multi_process_bfe)
    uda.set_bfe(bfe)
    algo = pg.algorithm(uda)

    # Setup population
    pop = pg.population(prob, size=args.optimization.population_size)

    # Evolve archipelago (Optimization process)
    algo.set_verbosity(1)
    pop = algo.evolve(pop)

    # Evolve (Optimization process)
    fitness_list = []
    for i in range(args.optimization.number_of_generations):
        pop = algo.evolve(pop)
        fitness_list.append(pop.get_f()[pop.best_idx()])
        print("Generations: ", i+1, " "*10, "Best: ", fitness_list[len(fitness_list)-1])

    # Logs for output
    run_time_end = time.time()
    run_time = run_time_end - run_time_start
    champion_f = pop.champion_f
    champion_x = pop.champion_x
    print("Optimization run time: ", run_time)
    print("Champion fitness value: ", champion_f) 
    print("Champion chromosome: ", champion_x) 

    # Shutdown pool to avoid mp_bfe bug for python==3.8
    multi_process_bfe.shutdown_pool()
    #multi_process_bfe.shutdown_view()

    return run_time, champion_f, champion_x, fitness_list


def main():

    # Setup problem parameters (as DotMaP)
    #args, lower_bounds, upper_bounds = setup_parameters()
    args = setup_parameters()
    
    # Setup initial state space
    initial_condition = [] #[1.02346115e+04, 5.04262474e-01, 1.40521347e+00, 3.03148072e+00, 2.68878957e-01, 5.74690265e+00]*args.problem.number_of_spacecrafts
    lower_bounds, upper_bounds = setup_initial_state_domain(initial_condition, 
                                                            args.problem.start_time, 
                                                            args.problem.final_time, 
                                                            args.problem.number_of_maneuvers, 
                                                            args.problem.number_of_spacecrafts)

    # Run optimization
    run_time, champion_f, champion_x, fitness_list = load_udp(args, initial_condition, lower_bounds, upper_bounds)

    # Save results
    run_time = np.asarray([float(run_time)])
    champion_f = np.asarray(champion_f)
    champion_x = np.asarray(champion_x)
    fitness_list = np.asarray(fitness_list)


    np.savetxt("run_time.csv", run_time, delimiter=",")
    np.savetxt("champion_f.csv", champion_f, delimiter=",")
    np.savetxt("champion_x.csv", champion_x, delimiter=",")
    np.savetxt("fitness_list.csv", fitness_list, delimiter=",")


if __name__ == "__main__":
    main()
    #print(pg.__version__)
    #cProfile.run("main()", "output.dat")

    #with open("output_time.txt", "w") as f:
    #    p = pstats.Stats("output.dat", stream=f)
    #    p.sort_stats("time").print_stats()
    
    #with open("output_calls.txt", "w") as f:
    #    p = pstats.Stats("output.dat", stream=f)
    #    p.sort_stats("calls").print_stats()
        