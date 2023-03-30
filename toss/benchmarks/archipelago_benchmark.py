# Core packages
import pygmo as pg
import numpy as np
import time
from math import pi
from dotmap import DotMap

# For creating mesh
import mesh_utility as mesh_utility

# For computing the next state
import equations_of_motion as equations_of_motion

# For optimization using pygmo
from udp_initial_condition import udp_initial_condition

# For parallellized initialization
import ipyparallel as ipp
from IPython.kernel import client



def setup_parameters():
    """Set up of required hyperparameters for the optimization scheme. 

    Returns:

        body_args (dotmap.DotMap): Parameters related to the celestial body:
            density (float): Body density of celestial body.
            mu (float): Gravitational parameter for celestial body.
            declination (float): Declination angle of spin axis.
            right_ascension (float): Right ascension angle of spin axis.
            spin_period (float): Rotational period around spin axis of the body.
            spin_velocity (float): Angular velocity of the body's rotation.
            spin_axis (np.ndarray): The axis around which the body rotates.

        integrator_args (dotmap.DotMap): Specific parameters related to the integrator:
            algorithm (int): Integer representing specific integrator algorithm.
            dense_output (bool): Dense output status of integrator.
            rtol (float): Relative error tolerance for integration.
            atol (float): Absolute error tolerance for integration.

        problem_args (dotmap.DotMap): Parameters related to the problem:
            start_time (float): Start time (in seconds) for the integration of trajectory.
            final_time (float): Final time (in seconds) for the integration of trajectory.
            initial_time_step (float): Size of initial time step (in seconds) for integration of trajectory.
            target_squared_altitude (float): Squared value of the satellite's orbital target altitude.
            radius_bounding_sphere (float): Radius of the bounding sphere representing risk zone for collisions with celestial body.
            event (int): Event configuration (0 = no event, 1 = collision with body detection)
        
        lower_bounds (np.ndarray): Lower boundary values for the initial state vector.
        upper_bounds (np.ndarray): Lower boundary values for the initial state vector.
                
        population_size (int): Number of chromosomes to compare at each generation.
        number_of_generations (int): Number of generations for the genetic opimization.
    """
    args = DotMap()

    # Setup body parameters
    args.body.density = 533                  # https://sci.esa.int/web/rosetta/-/14615-comet-67p
    args.body.mu = 665.666                   # Gravitational parameter for 67P/C-G
    args.body.declination = 64               # [degrees] https://sci.esa.int/web/rosetta/-/14615-comet-67p
    args.body.right_ascension = 69           # [degrees] https://sci.esa.int/web/rosetta/-/14615-comet-67p
    args.body.spin_period = 12.06*3600       # [seconds] https://sci.esa.int/web/rosetta/-/14615-comet-67p
    args.body.spin_velocity = (2*pi)/args.body.spin_period
    args.body.spin_axis = equations_of_motion.setup_spin_axis(args)

    # Setup specific integrator parameters:
    args.integrator.algorithm = 3
    args.integrator.dense_output = True
    args.integrator.rtol = 1e-12
    args.integrator.atol = 1e-12

    # Setup problem parameters
    args.problem.start_time = 0                     # Starting time [s]
    args.problem.final_time = 20*3600.0             # Final time [s]
    args.problem.initial_time_step = 600            # Initial time step size for integration [s]
    args.problem.radius_bounding_sphere = 4000      # Radius of spherical risk-zone for collision with celestial body [m]
    args.problem.target_squared_altitude = 8000**2  # Target altitude squared [m]
    args.problem.event = 1                          # Event configuration (0 = no event, 1 = collision with body detection)
    

    # Defining the parameter space for the optimization
    #   (Parameters are defined in osculating orbital elements)
    a = [5000, 15000] # Semi-major axis
    e = [0, 1]        # Eccentricity [0, 1]
    o = [0, 2*pi]     # Right ascension of ascending node [0,2*pi]
    w = [0, 2*pi]     # Argument of periapsis [0, 2*pi]
    i = [0, pi]       # Inclination [0, pi] 
    ea = [0, 2*pi]    # Eccentric anomaly [0, 2*pi]

    lower_bounds = np.array([a[0], e[0], i[0], o[0], w[0], ea[0]])
    upper_bounds = np.array([a[1], e[1], i[1], o[1], w[1], ea[1]])


    # Optimization parameters
    population_size = 7
    number_of_generations = 4
    number_of_islands = 4 # one per thread of the cpu

    return args, lower_bounds, upper_bounds, population_size, number_of_generations, number_of_islands

def pop_init(args, population_size, lower_bounds, upper_bounds):    
    
    # Setup User-Defined Problem (UDP)
    #print("Setting up the UDP...")
    udp = udp_initial_condition(args, lower_bounds, upper_bounds)
    prob = pg.problem(udp)

    # Setup optimization algorithm
    #print("Setting up the optimization algorithm...")
    assert population_size >= 7

    # Return population
    return pg.population(prob, population_size)


def find_solution(number_of_generations, number_of_islands, population_size):

    #print("Retrieving user defined parameters...")
    args, lower_bounds, upper_bounds, _, _, _ = setup_parameters()

    # Creating the mesh (TetGen)
    #print("Creating the mesh...")
    args.mesh.body, args.mesh.vertices, args.mesh.faces, args.mesh.largest_body_protuberant = mesh_utility.create_mesh()

    print("Setting up populations...")
    start_time = time.time()

    # Define populations for n number of islands
    cluster = ipp.Cluster(n=number_of_islands)
    cluster.start_cluster_sync()

    rc = ipp.Client()
    lview = rc.load_balanced_view()
    populations=list(lview.map(pop_init, [args]*number_of_islands, [population_size]*number_of_islands, [lower_bounds]*number_of_islands, [upper_bounds]*number_of_islands))
    rc.shutdown(hub=True)
    
    # Create Differential Evolution object by passing the number of generations as input
    uda = pg.sade(gen = number_of_generations)

    # Create pygmo algorithm object
    algo = pg.algorithm(uda)

    # Create empty archipelago and push each island onto it
    archi = pg.archipelago()
    for pop in populations:
        archi.push_back(algo=algo, pop=pop, udi=pg.ipyparallel_island())

    #mec = client.MultiEngineClient() 
    #mec.kill(controller=True)

    archi.set_topology(pg.fully_connected(len(archi)))
    archi.set_migration_type(pg.migration_type.broadcast)
    archi.evolve()
    print(archi)
    archi.wait()

    # Compute elapsed time for current optimzation process
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Get champion
    f_champion_per_island = archi.get_champions_f()
    x_champion_per_island = archi.get_champions_x()
    print("Champion fitness value: ", f_champion_per_island)
    print("Champion chromosome: ", x_champion_per_island)

    f_champion_idx = np.where(f_champion_per_island == min(f_champion_per_island))[0]
    x_champion = x_champion_per_island[f_champion_idx[0]]
    f_champion = f_champion_per_island[f_champion_idx[0]][0]

    return f_champion, x_champion, elapsed_time


def first_test():
    #Basic test to check that running script on Emma works:
    print("Initializing first test:")
    number_of_generations = 1
    number_of_islands = 10
    population_size = 7

    f_champion, x_champion, elapsed_time = find_solution(number_of_generations, number_of_islands, population_size)

    print("f champion: ", f_champion, "     x champion: ", x_champion, "     Elapsed time: ", elapsed_time)



def strong_scale_small():

    ######### Strong scaling: #########
    # Varying the number of threads (islands) used
    #  annd comparing results for a small and large run.
    islands = [1, 2, 4, 8, 16, 32]
    populations = [320, 160, 80, 40, 20, 10]

    #   small run:
    print("Now initializing strong scaling: small run")
    number_of_generations = 10
    small_run_strong_scaling = np.empty((8,len(islands)), dtype=np.float64) # (8xlen(islands)) array with each column representing: [f, x, n_islands] (Here: f:1x1,  x:6x1, t:1x1)
    for i in range(0,len(islands)):
        number_of_islands = islands[i]
        number_of_workers = number_of_islands
        population_size = populations[i]
        f_champion, x_champion, elapsed_time = find_solution(number_of_generations, number_of_islands, population_size)
        small_run_strong_scaling[0,i] = f_champion
        small_run_strong_scaling[1:7,i] = x_champion
        small_run_strong_scaling[7,i] = elapsed_time

        print("Elapsed time: ", elapsed_time)

    np.savetxt("small_run_strong_scaling.csv", small_run_strong_scaling, delimiter=",")


def strong_scale_large():
    islands = [1, 2, 4, 8, 16, 32] 
    populations = [320, 160, 80, 40, 20, 10]

    #   large run:
    print("Now initializing strong scaling: large run")
    number_of_generations = 32
    large_run_strong_scaling = np.empty((8,len(islands)), dtype=np.float64) # (8xlen(islands)) array with each column representing: [f, x, n_islands] (Here: f:1x1,  x:6x1, t:1x1)
    for i in range(0,len(islands)):
        number_of_islands = islands[i]
        number_of_workers = number_of_islands
        population_size = populations[i]
        f_champion, x_champion, elapsed_time = find_solution(number_of_generations, number_of_islands, population_size)
        large_run_strong_scaling[0,i] = f_champion
        large_run_strong_scaling[1:7,i] = x_champion
        large_run_strong_scaling[7,i] = elapsed_time

        print("Elapsed time: ", elapsed_time)

    np.savetxt("large_run_strong_scaling.csv", large_run_strong_scaling, delimiter=",")


def weak_scale():

    ######### Weak scaling: #########
    print("Now initializing weak scaling run.")
    # Scaling populations per thread (island) equivalent
    #  to the number threads used to see if performance
    #  is constants.
    islands =  [1, 2, 4, 8, 16, 32]
    number_of_generations = 32
    populations = [10, 10, 10, 10, 10, 10]

    weak_scaling = np.empty((8,len(islands)), dtype=np.float64) # (8xlen(islands)) array with each column representing: [f, x, n_islands] (Here: f:1x1,  x:6x1, t:1x1)
    for i in range(0,len(islands)):
        number_of_islands = islands[i]
        number_of_workers = number_of_islands
        population_size = populations[i]
        f_champion, x_champion, elapsed_time = find_solution(number_of_generations, number_of_islands, population_size)
        weak_scaling[0,i] = f_champion
        weak_scaling[1:7,i] = x_champion
        weak_scaling[7,i] = elapsed_time

        print("Elapsed time: ", elapsed_time)

    np.savetxt("weak_scaling.csv", weak_scaling, delimiter=",")


def run():

    # Initializing tests:
    first_test()

    #strong_scale_small()

    #strong_scale_large()

    #weak_scale()


if __name__ == "__main__":
    run()