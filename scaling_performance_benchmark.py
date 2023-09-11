# Core packages
import pygmo as pg
import cProfile
import pstats
import time
import numpy as np
import polyhedral_gravity


# Load required modules
from toss.optimization.udp_initial_condition import udp_initial_condition
from toss.optimization.setup_parameters import setup_parameters
from toss.optimization.setup_state import setup_initial_state_domain
from toss.fitness.fitness_function_utils import compute_space_coverage, update_spherical_tensor_grid
from toss.trajectory.compute_trajectory import compute_trajectory
from toss.trajectory.trajectory_tools import get_trajectory_fixed_step
from toss.trajectory.equations_of_motion import compute_motion

#import logging.config
#logging.config.dictConfig({
#'version': 1,
#'disable_existing_loggers': True})

import logging
#logging.basicConfig(level=logging.CRITICAL)
#logging.getLogger('polyhedral_gravity').setLevel(logging.CRITICAL)
#logger.propagate = False
#logging.getLogger("polyhedral_gravity").setLevel(logging.CRITICAL)
#logging.Logger.disabled = True

#logger = logging.getLogger('polyhedral_gravity')
#logging.basicConfig(level=logging.CRITICAL, filename='logfile.log')

logging.Logger.manager.loggerDict
logging.getLogger('polyhedral_gravity').disabled = True


def load_udp(args, initial_state, lower_bounds, upper_bounds):
    """Loads the provided user-defined problem (UDP).

    Args:
        args (dotmap.DotMap): A dotmap consisting of parameters related to the UDP.
        initial_state (np.ndarray): Initial state for the spacecraft (Non-empty array when given initial position or initial velocity).
        lower_bounds (np.ndarray): Lower bound for the chromosome (independent variables).
        upper_bounds (np.ndarray): Lower bound for the chromosome (independent variables).

    Returns:
        prob (object): A UDP-class suitable for PyGMO.
    """

    # Setup User-Defined Problem (UDP)
    print("Setting up the UDP...")
    udp = udp_initial_condition(args, initial_state, lower_bounds, upper_bounds)
    prob = pg.problem(udp)

    return prob


def load_uda(args, bfe):
    """Cretes the user-defined algorithm (UDA) with assigned parameters.

    Args:
        args (dotmap.DotMap): A dotmap consisting of parameters related to the UDA.
        bfe (object): Batch Fitness Evaluator 

    Returns:
        algo (object): A UDA-class suitable for PyGMO.
    """
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
    
    # Define the UDA.
    uda.set_bfe(bfe)
    algo = pg.algorithm(uda)
    
    return algo


def run_optimization(args, initial_state, lower_bounds, upper_bounds):
    """
    Main optimization script. Runs the module to optimize the chromosome for each 
    spacecraft at the time, where the spherical tensor used for computing coverage
    is updated amd passed in a feedback loop for trajectory optimization of the following spacecraft.

    Args:
        args (dotmap.DotMap): A dotmap consisting of parameters related to the optimization (eg mesh, body etc).
        initial_state (np.ndarray): Initial state vector for each spacecraft (Non-empty arrays when given initial position or initial velocity).
        lower_bounds (np.ndarray): Lower bound for the chromosome (independent variables).
        upper_bounds (np.ndarray): Lower bound for the chromosome (independent variables).

    Returns:
        run_time (float): Runtime for the complete optimization process.
        champion_f_array (np.ndarray): Champion fitness value for each spacecraft trajectory. 
        champion_x_array (np.ndarray): Champion chromosome related to each spacecraft.
        fitness_arr (np.ndarray): (N_Spacecraft x N_GenFitness) Array of fitness values for each generation corresonding to the optimization of each chromosome.
    """

    # Setup BFE machinery for paralellization (activates engines)
    multi_process_bfe = pg.mp_bfe() #pg.ipyparallel_bfe() #n=args.optimization.number_of_threads
    multi_process_bfe.resize_pool(args.optimization.number_of_threads)
    bfe = pg.bfe(multi_process_bfe)

    # Setup arrays for storing results:
    champion_f_list = []
    champion_x_list = []
    fitness_array = np.empty((args.problem.number_of_spacecrafts, args.optimization.number_of_generations), dtype=np.float64)

    # Initiate timer of the optimization process
    timer_start = time.time()
    
    # Optimize each spacecraft trajectory
    for spacecraft_i in range(0, args.problem.number_of_spacecrafts):

        # Setup udp
        if len(initial_state) == 0:
            prob = load_udp(args, [], lower_bounds, upper_bounds)
        else:
            prob = load_udp(args, initial_state[spacecraft_i], lower_bounds, upper_bounds)

        # Setup population
        pop = pg.population(prob=prob, size=args.optimization.population_size, b=bfe)

        # Setup uda
        algo = load_uda(args, bfe)

        # Evolve population
        algo.set_verbosity(1)
        pop = algo.evolve(pop)

        # Store champion fitness of each generation:
        optimization_info = algo.extract(pg.gaco).get_log()
        fitness_list = np.empty(args.optimization.number_of_generations,dtype=np.float64)
        for generation, info in enumerate(optimization_info):
            fitness_list[generation] = info[2]

        # Other logs for output
        champion_f = pop.champion_f
        champion_x = pop.champion_x

        # Recompute trajectory from champion chromosome.
        # NOTE: For a detailed description on constants required in dotmap args,
        #       see docstring for the function: compute_trajectory()
        #if len(initial_state[spacecraft_i]) > 0:
        #    state_vector = np.hstack((initial_state[spacecraft_i], champion_x))
        #else:
        #    state_vector = champion_x
        state_vector = champion_x

        _, list_of_ode_objects, _ = compute_trajectory(state_vector, args, compute_motion)
        positions, velocities, timesteps = get_trajectory_fixed_step(args, list_of_ode_objects)
        
        # Update boolean tensor (using trajectory resulting from champion chromosome)
        args.problem.bool_tensor = update_spherical_tensor_grid(args.problem.number_of_spacecrafts, args.body.spin_axis, args.body.spin_velocity, positions, velocities, timesteps, args.problem.radius_inner_bounding_sphere, args.problem.radius_outer_bounding_sphere, args.problem.tensor_grid_r, args.problem.tensor_grid_theta, args.problem.tensor_grid_phi, args.problem.weight_tensor, args.body.quaternion_rotation_objects)

        # Store champion information
        champion_f_list.append(champion_f)
        champion_x_list.append(champion_x)
        fitness_array[spacecraft_i, :] = fitness_list

    # Compute complete optimization run time.
    timer_end = time.time()
    run_time = timer_end - timer_start

    # Shutdown pool to avoid mp_bfe bug for python==3.8
    multi_process_bfe.shutdown_pool()

    return run_time, champion_f_list, champion_x_list, fitness_array




def scaling(generations, threads, populations):
    """
    Generates results for a strong scaling test.
    """
    args = setup_parameters()

    # Scaling results: 
    #   n_rows = 2 + n_spacecraft*(n_initial_arg + 5*n_maneuvers)
    scaling_results = np.empty((14,len(threads)), dtype=np.float64)
    for i in range(0, len(threads)):

        # Setup optimization parameters:
        args.optimization.number_of_threads = threads[i]
        args.optimization.population_size = populations[i]
        args.optimization.number_of_generations = generations[i]


        # Setup initial state
        initial_state = [] #np.array_split([args.problem.initial_x, args.problem.initial_y, args.problem.initial_z]*args.problem.number_of_spacecrafts, args.problem.number_of_spacecrafts)

        # Setup boundary constraints for the chromosome   NOTE: initial_state[0]
        args.chromosome.x_min = 4000
        args.chromosome.x_max = 12500
        args.chromosome.y_min = -np.pi/2
        args.chromosome.y_max = np.pi/2
        args.chromosome.z_min = 0
        args.chromosome.z_max = 2*np.pi

        lower_bounds, upper_bounds = setup_initial_state_domain(initial_state, 
                                                                args.problem.start_time, 
                                                                args.problem.final_time, 
                                                                args.problem.number_of_maneuvers, 
                                                                args.problem.number_of_spacecrafts,
                                                                args.chromosome)

        # Run optimization
        run_time, champion_f, champion_x, _ = run_optimization(args, initial_state, lower_bounds, upper_bounds)

        # Store results
        scaling_results[0,i] = champion_f[0]
        scaling_results[1:13,i] = np.asarray(champion_x)
        scaling_results[13,i] = run_time

        # For logs:
        print("Threads: ", args.optimization.number_of_threads, "   Pop: ", args.optimization.population_size,  "   Gen: ", args.optimization.number_of_generations)
        print("f: ", champion_f, "   elapsed time: ", run_time)

    return scaling_results



def run_scaling_benchmark():

    #polyhedral_gravity.LOGGING_LEVEL = 5

    # Initial test:
    generations = [100]
    threads = [200]
    populations = [256]
    initial_test_results = scaling(generations, threads, populations)
    np.savetxt("new_initial_test_results.csv", initial_test_results, delimiter=",")

    # Strong scaling, small run:
    generations = [10, 10, 10, 10, 10, 10, 10, 10, 10]
    threads = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    populations = [256, 256, 256, 256, 256, 256, 256, 256, 256]
    strong_scaling_results = scaling(generations, threads, populations)
    np.savetxt("new_strong_scaling_small_run__1S_1M_12H.csv", strong_scaling_results, delimiter=",")

    # Strong scaling, large run:
    generations = [10, 10, 10, 10, 10, 10, 10, 10, 10]
    threads = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    populations = [512, 512, 512, 512, 512, 512, 512, 512, 512]
    strong_scaling_results = scaling(generations, threads, populations)
    np.savetxt("new_strong_scaling_large_run__1S_1M_12H.csv", strong_scaling_results, delimiter=",")

    # Weak scaling
    generations = [10, 10, 10, 10, 10, 10, 10, 10, 10]
    threads =  [1, 2, 4, 8, 16, 32, 64, 128, 256]
    populations = [20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
    weak_scaling_results = scaling(generations, threads, populations)
    np.savetxt("new_weak_scaling__1S_1M_12H.csv", weak_scaling_results, delimiter=",")


if __name__ == "__main__":
    run_scaling_benchmark()    