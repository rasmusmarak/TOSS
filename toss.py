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
from toss.fitness.fitness_function_utils import compute_space_coverage, update_spherical_tensor_grid
from toss.trajectory.compute_trajectory import compute_trajectory
from toss.trajectory.trajectory_tools import get_trajectory_fixed_step
from toss.trajectory.equations_of_motion import compute_motion


def load_udp(args, initial_condition, lower_bounds, upper_bounds):
    """Loads the provided user-defined problem (UDP).

    Args:
        args (dotmap.DotMap): A dotmap consisting of parameters related to the UDP.
        initial_condition (np.ndarray): Initial condition for the spacecraft (Non-empty array when given initial position or initial velocity).
        lower_bounds (np.ndarray): Lower bound for the chromosome (independent variables).
        upper_bounds (np.ndarray): Lower bound for the chromosome (independent variables).

    Returns:
        prob (object): A UDP-class suitable for PyGMO.
    """

    # Setup User-Defined Problem (UDP)
    print("Setting up the UDP...")
    udp = udp_initial_condition(args, initial_condition, lower_bounds, upper_bounds)
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
    champion_f_array = np.empty(args.problem.number_of_spacecrafts, dtype=np.float64)
    champion_x_array = np.empty(args.problem.number_of_spacecrafts*len(lower_bounds), dtype=np.float64)
    fitness_array = np.empty((args.problem.number_of_spacecrafts, args.optimization.number_of_generations), dtype=np.float64)

    # Initiate timer of the optimization process
    timer_start = time.time()
    
    # Optimize each spacecraft trajectory
    for spacecraft_i in range(0, args.problem.number_of_spacecrafts):

        # Setup udp
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

        # Update boolean tensor (using trajectory resulting from champion chromosome)
        if len(initial_state[spacecraft_i]) > 0:
            state_vector = np.hstack((initial_state[spacecraft_i], champion_x))
        else:
            state_vector = champion_x

        _, list_of_ode_objects, _ = compute_trajectory(state_vector, args, compute_motion)
        positions, velocities, timesteps = get_trajectory_fixed_step(args, list_of_ode_objects)
        args.problem.bool_tensor = update_spherical_tensor_grid(args.problem.number_of_spacecrafts, args.body.spin_axis, args.body.spin_velocity, positions, velocities, timesteps, args.problem.radius_inner_bounding_sphere, args.problem.radius_outer_bounding_sphere, args.problem.tensor_grid_r, args.problem.tensor_grid_theta, args.problem.tensor_grid_phi, args.problem.bool_tensor)

        # Store champion information
        champion_f_array[spacecraft_i] = champion_f
        champion_x_array[len(champion_x)*spacecraft_i:len(champion_x)*(spacecraft_i + 1)] = champion_x
        fitness_array[spacecraft_i, :] = fitness_list

    # Compute complete optimization run time.
    timer_end = time.time()
    run_time = timer_end - timer_start

    # Shutdown pool to avoid mp_bfe bug for python==3.8
    multi_process_bfe.shutdown_pool()

    return run_time, champion_f_array, champion_x_array, fitness_array


def main():
    """ 
    Main function. Defines parameters, domain and initial condition and then calls the main optimization script.
    The results are stored in corresponding csv-files. 
    """
    # Setup problem parameters (as DotMaP)
    args = setup_parameters()
    
    # Setup initial state (eg position)
    initial_state = np.array_split([-135.13402075, -4089.53592604, 6050.17636635]*args.problem.number_of_spacecrafts, args.problem.number_of_spacecrafts) # [-3645.61233166, -5634.25158292, 2883.0399209] # [-135.13402075, -4089.53592604, 6050.17636635], [-135.13402075, -4089.53592604, 6050.17636635]

    # Setup boundary constraints for the chromosome
    lower_bounds, upper_bounds = setup_initial_state_domain(initial_state[0], 
                                                            args.problem.start_time, 
                                                            args.problem.final_time, 
                                                            args.problem.number_of_maneuvers, 
                                                            args.problem.number_of_spacecrafts,
                                                            args.chromosome)

    # Run optimization
    run_time, champion_f, champion_x, fitness_list = run_optimization(args, initial_state, lower_bounds, upper_bounds)

    # Save results
    run_time = np.asarray([float(run_time)])
    champion_f = np.asarray(champion_f)
    champion_x = np.asarray(champion_x)
    fitness_list = np.asarray(fitness_list)
    np.savetxt("test_run_time.csv", run_time, delimiter=",")
    np.savetxt("test_champion_f.csv", champion_f, delimiter=",")
    np.savetxt("test_champion_x.csv", champion_x, delimiter=",")
    np.savetxt("test_fitness_list.csv", fitness_list, delimiter=",")



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
        