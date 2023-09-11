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
from toss.fitness.fitness_function_utils import compute_space_coverage, update_spherical_tensor_grid,sphere2cart
from toss.trajectory.compute_trajectory import compute_trajectory
from toss.trajectory.trajectory_tools import get_trajectory_fixed_step
from toss.trajectory.equations_of_motion import compute_motion
from toss.fitness.fitness_function_enums import FitnessFunctions
from toss.fitness.fitness_functions import get_fitness

import logging
#logging.basicConfig(level=logging.CRITICAL)
#logging.getLogger().setLevel(logging.CRITICAL)
#logging.getLogger("polyhedral_gravity").setLevel(logging.CRITICAL)

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
        args.algorithm.memory_parameter,
        args.algorithm.seed)
    
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

    # Initiate timer of the optimization process
    timer_start = time.time()

    # Setup udp
    if len(initial_state) == 0:
        prob = load_udp(args, [], lower_bounds, upper_bounds)
    else:
        prob = load_udp(args, initial_state, lower_bounds, upper_bounds)

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

    # Compute complete optimization run time.
    timer_end = time.time()
    run_time = timer_end - timer_start

    # Recompute trajectory from champion chromosome.
    # NOTE: For a detailed description on constants required in dotmap args,
    #       see docstring for the function: compute_trajectory()

    if len(initial_state) == 0:
        x,y,z = sphere2cart(champion_x[0], champion_x[1], champion_x[2])
        champion_x[0] = x
        champion_x[1] = y
        champion_x[2] = z
        spacecraft_info = champion_x
    else:
        spacecraft_info = np.hstack((initial_state, champion_x))
    # Compute Trajectory and resample for a given fixed time-step delta t        
    collision_detected, list_of_ode_objects, _ = compute_trajectory(spacecraft_info, args, compute_motion)

    # Resample trajectory for a fixed time-step delta t
    positions, velocities, timesteps = get_trajectory_fixed_step(args, list_of_ode_objects)
            
    # Compute aggregate fitness:
    chosen_fitness_function = FitnessFunctions.CoveredSpaceCloseDistancePenaltyFarDistancePenalty
    fitness = get_fitness(chosen_fitness_function, args, positions, velocities, timesteps)
    print("Champion fitness: ", fitness)

    # Shutdown pool to avoid mp_bfe bug for python==3.8
    multi_process_bfe.shutdown_pool()

    return run_time, champion_f, champion_x, fitness_list


def main(args):
    """ 
    Main function. Defines parameters, domain and initial condition and then calls the main optimization script.
    The results are stored in corresponding csv-files. 
    """
    #args = setup_parameters()
    
    # Setup initial state
    # NOTE Initial state can be varied dependent on what is optimized, eg:
    #       []
    #       [position]
    #       [position, velocity]
    initial_state = [] #np.array_split([args.problem.initial_x, args.problem.initial_y, args.problem.initial_z]*args.problem.number_of_spacecrafts, args.problem.number_of_spacecrafts)

    # Setup boundary constraints for the chromosome   NOTE: initial_state[0]
    lower_bounds, upper_bounds = setup_initial_state_domain(initial_state, 
                                                            args.problem.start_time, 
                                                            args.problem.final_time, 
                                                            args.problem.number_of_maneuvers, 
                                                            args.problem.number_of_spacecrafts,
                                                            args.chromosome)

    lower_bounds = list(lower_bounds)*args.problem.number_of_spacecrafts
    upper_bounds = list(upper_bounds)*args.problem.number_of_spacecrafts

    # Run optimization
    run_time, champion_f, champion_x, fitness_list = run_optimization(args, initial_state, lower_bounds, upper_bounds)

    # Save results
    run_time = np.asarray([float(run_time)])
    champion_f = np.asarray(champion_f)
    champion_x = np.asarray(champion_x)
    fitness_list = np.asarray(fitness_list)

    return run_time, champion_f, champion_x, fitness_list
    
    #np.savetxt(run_id + "run_time.csv", run_time, delimiter=",")
    #np.savetxt(run_id + "champion_f.csv", champion_f, delimiter=",")
    #np.savetxt(run_id + "champion_x.csv", champion_x, delimiter=",")
    #np.savetxt(run_id + "fitness_list.csv", fitness_list, delimiter=",")    

def old_functions():
    # Test: kernal
    args = setup_parameters()
    ker_cases = [10, 50, 100, 150, 200]
    ker_list_t = np.empty((3, len(ker_cases)), dtype=np.float64)
    ker_list_f = np.empty((3, len(ker_cases)), dtype=np.float64)
    ker_list_f_list_0 = np.empty((args.optimization.number_of_generations, len(ker_cases)), dtype=np.float64)
    ker_list_f_list_1 = np.empty((args.optimization.number_of_generations, len(ker_cases)), dtype=np.float64)
    ker_list_f_list_2 = np.empty((args.optimization.number_of_generations, len(ker_cases)), dtype=np.float64)
    for j in [0,1,2]:
        for i, ker in enumerate(ker_cases):
            args.algorithm.kernel_size = ker
            t, f, x, f_list = main(args)
            ker_list_t[j,i] = t
            ker_list_f[j,i] = f

            if j==0:
                ker_list_f_list_0[:,i] = f_list
            elif j==1:
                ker_list_f_list_1[:,i] = f_list
            else:
                ker_list_f_list_2[:,i] = f_list
    np.savetxt("ker_tests_t.csv", np.asarray(ker_list_t), delimiter=",")
    np.savetxt("ker_tests_f.csv", np.asarray(ker_list_f), delimiter=",")
    np.savetxt("ker_tests_f_list_0.csv", ker_list_f_list_0, delimiter=",")
    np.savetxt("ker_tests_f_list_1.csv", ker_list_f_list_1, delimiter=",")
    np.savetxt("ker_tests_f_list_2.csv", ker_list_f_list_2, delimiter=",")
    

    # Test: accuracy_parameter
    args = setup_parameters()
    acc_cases = [0, 0.01, 0.1]
    acc_list_t = np.empty((3, len(acc_cases)), dtype=np.float64)
    acc_list_f = np.empty((3, len(acc_cases)), dtype=np.float64)
    acc_list_f_list_0 = np.empty((args.optimization.number_of_generations, len(acc_cases)), dtype=np.float64)
    acc_list_f_list_1 = np.empty((args.optimization.number_of_generations, len(acc_cases)), dtype=np.float64)
    acc_list_f_list_2 = np.empty((args.optimization.number_of_generations, len(acc_cases)), dtype=np.float64)    
    for j in [0,1,2]:
        for i, acc in enumerate(acc_cases):
            args.algorithm.accuracy_parameter = acc
            t, f, x, f_list = main(args)
            acc_list_t[j,i] = t
            acc_list_f[j,i] = f
            
            if j==0:
                acc_list_f_list_0[:,i] = f_list
            elif j==1:
                acc_list_f_list_1[:,i] = f_list
            else:
                acc_list_f_list_2[:,i] = f_list
    np.savetxt("acc_tests_t.csv", np.asarray(acc_list_t), delimiter=",")
    np.savetxt("acc_tests_f.csv", np.asarray(acc_list_f), delimiter=",")
    np.savetxt("acc_list_f_list_0.csv", acc_list_f_list_0, delimiter=",")
    np.savetxt("acc_list_f_list_1.csv", acc_list_f_list_1, delimiter=",")
    np.savetxt("acc_list_f_list_2.csv", acc_list_f_list_2, delimiter=",")

    # Test: convergence_speed_parameter
    args = setup_parameters()
    c_speed_cases = [0, 0.5, 1, 2]
    c_speed_list_t = np.empty((3, len(c_speed_cases)), dtype=np.float64)
    c_speed_list_f = np.empty((3, len(c_speed_cases)), dtype=np.float64)
    c_speed_list_f_list_0 = np.empty((args.optimization.number_of_generations, len(c_speed_cases)), dtype=np.float64)
    c_speed_list_f_list_1 = np.empty((args.optimization.number_of_generations, len(c_speed_cases)), dtype=np.float64)
    c_speed_list_f_list_2 = np.empty((args.optimization.number_of_generations, len(c_speed_cases)), dtype=np.float64) 
    for j in [0,1,2]:
        for i, c_speed in enumerate(c_speed_cases):
            args.algorithm.convergence_speed_parameter = c_speed
            t, f, x, f_list = main(args)
            c_speed_list_t[j,i] = t
            c_speed_list_f[j,i] = f
            if j==0:
                c_speed_list_f_list_0[:,i] = f_list
            elif j==1:
                c_speed_list_f_list_1[:,i] = f_list
            else:
                c_speed_list_f_list_2[:,i] = f_list
    np.savetxt("c_speed_tests_t.csv", np.asarray(c_speed_list_t), delimiter=",")
    np.savetxt("c_speed_tests_f.csv", np.asarray(c_speed_list_f), delimiter=",")
    np.savetxt("c_speed_list_f_list_0.csv", c_speed_list_f_list_0, delimiter=",")
    np.savetxt("c_speed_list_f_list_1.csv", c_speed_list_f_list_1, delimiter=",")
    np.savetxt("c_speed_list_f_list_2.csv", c_speed_list_f_list_2, delimiter=",")

    # Test: threshold_parameter
    args = setup_parameters()
    thresh_cases = [1, 25, 50]
    thresh_list_t = np.empty((3, len(thresh_cases)), dtype=np.float64)
    thresh_list_f = np.empty((3, len(thresh_cases)), dtype=np.float64)
    thresh_list_f_list_0 = np.empty((args.optimization.number_of_generations, len(thresh_cases)), dtype=np.float64)
    thresh_list_f_list_1 = np.empty((args.optimization.number_of_generations, len(thresh_cases)), dtype=np.float64)
    thresh_list_f_list_2 = np.empty((args.optimization.number_of_generations, len(thresh_cases)), dtype=np.float64) 
    for j in [0,1,2]:
        for i, thresh in enumerate(thresh_cases):
            args.algorithm.threshold_parameter = thresh
            t, f, x, f_list = main(args)
            thresh_list_t[j,i] = t
            thresh_list_f[j,i] = f
            if j==0:
                thresh_list_f_list_0[:,i] = f_list
            elif j==1:
                thresh_list_f_list_1[:,i] = f_list
            else:
                thresh_list_f_list_2[:,i] = f_list
    np.savetxt("thresh_tests_t.csv", np.asarray(thresh_list_t), delimiter=",")
    np.savetxt("thresh_tests_f.csv", np.asarray(thresh_list_f), delimiter=",")
    np.savetxt("thresh_list_f_list_0.csv", thresh_list_f_list_0, delimiter=",")
    np.savetxt("thresh_list_f_list_1.csv", thresh_list_f_list_1, delimiter=",")
    np.savetxt("thresh_list_f_list_2.csv", thresh_list_f_list_2, delimiter=",")

    # Test: std_convergence_speed_parameter
    args = setup_parameters()
    Nmark_cases = [7, 50, 100, 200]
    Nmark_list_t = np.empty((3, len(Nmark_cases)), dtype=np.float64)
    Nmark_list_f = np.empty((3, len(Nmark_cases)), dtype=np.float64)
    Nmark_list_f_list_0 = np.empty((args.optimization.number_of_generations, len(Nmark_cases)), dtype=np.float64)
    Nmark_list_f_list_1 = np.empty((args.optimization.number_of_generations, len(Nmark_cases)), dtype=np.float64)
    Nmark_list_f_list_2 = np.empty((args.optimization.number_of_generations, len(Nmark_cases)), dtype=np.float64) 
    for j in [0,1,2]:
        for i, Nmark in enumerate(Nmark_cases):
            args.algorithm.std_convergence_speed_parameter = Nmark
            t, f, x, f_list = main(args)
            Nmark_list_t[j,i] = t
            Nmark_list_f[j,i] = f
            if j==0:
                Nmark_list_f_list_0[:,i] = f_list
            elif j==1:
                Nmark_list_f_list_1[:,i] = f_list
            else:
                Nmark_list_f_list_2[:,i] = f_list
    np.savetxt("Nmark_tests_t.csv", np.asarray(Nmark_list_t), delimiter=",")
    np.savetxt("Nmark_tests_f.csv", np.asarray(Nmark_list_f), delimiter=",")
    np.savetxt("Nmark_list_f_list_0.csv", Nmark_list_f_list_0, delimiter=",")
    np.savetxt("Nmark_list_f_list_1.csv", Nmark_list_f_list_1, delimiter=",")
    np.savetxt("Nmark_list_f_list_2.csv", Nmark_list_f_list_2, delimiter=",")




if __name__ == "__main__":

    seeds = [76237, 56436, 2049]

    # Test: kernal
    args = setup_parameters()
    ker_cases = [10, 50, 100, 150, 200]
    ker_list_t = np.empty((3, len(ker_cases)), dtype=np.float64)
    ker_list_f = np.empty((3, len(ker_cases)), dtype=np.float64)
    ker_list_f_list_0 = np.empty((args.optimization.number_of_generations, len(ker_cases)), dtype=np.float64)
    ker_list_f_list_1 = np.empty((args.optimization.number_of_generations, len(ker_cases)), dtype=np.float64)
    ker_list_f_list_2 = np.empty((args.optimization.number_of_generations, len(ker_cases)), dtype=np.float64)
    for j in [0,1,2]:
        for i, ker in enumerate(ker_cases):
            args.algorithm.kernel_size = ker
            args.algorithm.seed = seeds[j]
            t, f, x, f_list = main(args)
            ker_list_t[j,i] = t
            ker_list_f[j,i] = f

            if j==0:
                ker_list_f_list_0[:,i] = f_list
            elif j==1:
                ker_list_f_list_1[:,i] = f_list
            else:
                ker_list_f_list_2[:,i] = f_list
    np.savetxt("ker_tests_t.csv", np.asarray(ker_list_t), delimiter=",")
    np.savetxt("ker_tests_f.csv", np.asarray(ker_list_f), delimiter=",")
    np.savetxt("ker_tests_f_list_0.csv", ker_list_f_list_0, delimiter=",")
    np.savetxt("ker_tests_f_list_1.csv", ker_list_f_list_1, delimiter=",")
    np.savetxt("ker_tests_f_list_2.csv", ker_list_f_list_2, delimiter=",")
    

    # Test: accuracy_parameter
    args = setup_parameters()
    acc_cases = [0, 0.01]
    acc_list_t = np.empty((3, len(acc_cases)), dtype=np.float64)
    acc_list_f = np.empty((3, len(acc_cases)), dtype=np.float64)
    acc_list_f_list_0 = np.empty((args.optimization.number_of_generations, len(acc_cases)), dtype=np.float64)
    acc_list_f_list_1 = np.empty((args.optimization.number_of_generations, len(acc_cases)), dtype=np.float64)
    acc_list_f_list_2 = np.empty((args.optimization.number_of_generations, len(acc_cases)), dtype=np.float64)    
    for j in [0,1,2]:
        for i, acc in enumerate(acc_cases):
            args.algorithm.accuracy_parameter = acc
            args.algorithm.seed = seeds[j]
            t, f, x, f_list = main(args)
            acc_list_t[j,i] = t
            acc_list_f[j,i] = f
            
            if j==0:
                acc_list_f_list_0[:,i] = f_list
            elif j==1:
                acc_list_f_list_1[:,i] = f_list
            else:
                acc_list_f_list_2[:,i] = f_list
    np.savetxt("acc_tests_t.csv", np.asarray(acc_list_t), delimiter=",")
    np.savetxt("acc_tests_f.csv", np.asarray(acc_list_f), delimiter=",")
    np.savetxt("acc_list_f_list_0.csv", acc_list_f_list_0, delimiter=",")
    np.savetxt("acc_list_f_list_1.csv", acc_list_f_list_1, delimiter=",")
    np.savetxt("acc_list_f_list_2.csv", acc_list_f_list_2, delimiter=",")

    # Test: convergence_speed_parameter
    args = setup_parameters()
    c_speed_cases = [0, 0.5, 1, 2]
    c_speed_list_t = np.empty((3, len(c_speed_cases)), dtype=np.float64)
    c_speed_list_f = np.empty((3, len(c_speed_cases)), dtype=np.float64)
    c_speed_list_f_list_0 = np.empty((args.optimization.number_of_generations, len(c_speed_cases)), dtype=np.float64)
    c_speed_list_f_list_1 = np.empty((args.optimization.number_of_generations, len(c_speed_cases)), dtype=np.float64)
    c_speed_list_f_list_2 = np.empty((args.optimization.number_of_generations, len(c_speed_cases)), dtype=np.float64) 
    for j in [0,1,2]:
        for i, c_speed in enumerate(c_speed_cases):
            args.algorithm.convergence_speed_parameter = c_speed
            args.algorithm.seed = seeds[j]
            t, f, x, f_list = main(args)
            c_speed_list_t[j,i] = t
            c_speed_list_f[j,i] = f
            if j==0:
                c_speed_list_f_list_0[:,i] = f_list
            elif j==1:
                c_speed_list_f_list_1[:,i] = f_list
            else:
                c_speed_list_f_list_2[:,i] = f_list
    np.savetxt("c_speed_tests_t.csv", np.asarray(c_speed_list_t), delimiter=",")
    np.savetxt("c_speed_tests_f.csv", np.asarray(c_speed_list_f), delimiter=",")
    np.savetxt("c_speed_list_f_list_0.csv", c_speed_list_f_list_0, delimiter=",")
    np.savetxt("c_speed_list_f_list_1.csv", c_speed_list_f_list_1, delimiter=",")
    np.savetxt("c_speed_list_f_list_2.csv", c_speed_list_f_list_2, delimiter=",")

    # Test: threshold_parameter
    args = setup_parameters()
    thresh_cases = [1, 25, 50, 100, 250, 500, 1000]
    thresh_list_t = np.empty((3, len(thresh_cases)), dtype=np.float64)
    thresh_list_f = np.empty((3, len(thresh_cases)), dtype=np.float64)
    thresh_list_f_list_0 = np.empty((args.optimization.number_of_generations, len(thresh_cases)), dtype=np.float64)
    thresh_list_f_list_1 = np.empty((args.optimization.number_of_generations, len(thresh_cases)), dtype=np.float64)
    thresh_list_f_list_2 = np.empty((args.optimization.number_of_generations, len(thresh_cases)), dtype=np.float64) 
    for j in [0,1,2]:
        for i, thresh in enumerate(thresh_cases):
            args.algorithm.threshold_parameter = thresh
            args.algorithm.seed = seeds[j]
            t, f, x, f_list = main(args)
            thresh_list_t[j,i] = t
            thresh_list_f[j,i] = f
            if j==0:
                thresh_list_f_list_0[:,i] = f_list
            elif j==1:
                thresh_list_f_list_1[:,i] = f_list
            else:
                thresh_list_f_list_2[:,i] = f_list
    np.savetxt("thresh_tests_t.csv", np.asarray(thresh_list_t), delimiter=",")
    np.savetxt("thresh_tests_f.csv", np.asarray(thresh_list_f), delimiter=",")
    np.savetxt("thresh_list_f_list_0.csv", thresh_list_f_list_0, delimiter=",")
    np.savetxt("thresh_list_f_list_1.csv", thresh_list_f_list_1, delimiter=",")
    np.savetxt("thresh_list_f_list_2.csv", thresh_list_f_list_2, delimiter=",")

    # Test: std_convergence_speed_parameter
    args = setup_parameters()
    Nmark_cases = [7, 50, 100, 200]
    Nmark_list_t = np.empty((3, len(Nmark_cases)), dtype=np.float64)
    Nmark_list_f = np.empty((3, len(Nmark_cases)), dtype=np.float64)
    Nmark_list_f_list_0 = np.empty((args.optimization.number_of_generations, len(Nmark_cases)), dtype=np.float64)
    Nmark_list_f_list_1 = np.empty((args.optimization.number_of_generations, len(Nmark_cases)), dtype=np.float64)
    Nmark_list_f_list_2 = np.empty((args.optimization.number_of_generations, len(Nmark_cases)), dtype=np.float64) 
    for j in [0,1,2]:
        for i, Nmark in enumerate(Nmark_cases):
            args.algorithm.std_convergence_speed_parameter = Nmark
            args.algorithm.seed = seeds[j]
            t, f, x, f_list = main(args)
            Nmark_list_t[j,i] = t
            Nmark_list_f[j,i] = f
            if j==0:
                Nmark_list_f_list_0[:,i] = f_list
            elif j==1:
                Nmark_list_f_list_1[:,i] = f_list
            else:
                Nmark_list_f_list_2[:,i] = f_list
    np.savetxt("Nmark_tests_t.csv", np.asarray(Nmark_list_t), delimiter=",")
    np.savetxt("Nmark_tests_f.csv", np.asarray(Nmark_list_f), delimiter=",")
    np.savetxt("Nmark_list_f_list_0.csv", Nmark_list_f_list_0, delimiter=",")
    np.savetxt("Nmark_list_f_list_1.csv", Nmark_list_f_list_1, delimiter=",")
    np.savetxt("Nmark_list_f_list_2.csv", Nmark_list_f_list_2, delimiter=",")




