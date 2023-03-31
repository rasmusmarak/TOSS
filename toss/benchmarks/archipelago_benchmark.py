# Core packages
import pygmo as pg
import numpy as np
import time

# For optimization using pygmo
#from udp_initial_condition import udp_initial_condition

# For parallellized initialization
import ipyparallel as ipp
#from IPython.kernel import client

import benchmarks.pop_initial as pop_initial



def find_solution(number_of_generations, number_of_islands, population_size):

    print("Setting up populations...")
    start_time = time.time()

    # Define populations for n number of islands
    #cluster = ipp.Cluster(n=number_of_islands)
    #cluster.start_cluster_sync()

    rc = ipp.Client()
    lview = rc.load_balanced_view()

    populations=list(lview.map(pop_initial.pop_init, [population_size]*number_of_islands))
    rc.shutdown(hub=True)
    
    # Create Differential Evolution object by passing the number of generations as input
    uda = pg.sade(gen = number_of_generations)

    # Create pygmo algorithm object
    algo = pg.algorithm(uda)

    # Create empty archipelago and push each island onto it
    archi = pg.archipelago()
    for pop in populations:
        archi.push_back(algo=algo, pop=pop, udi=pg.ipyparallel_island())
    
    mid_time = time.time()

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

    print("init time: ", mid_time-start_time)
    print("evolve time: ", end_time-mid_time)
    print("total time: ", elapsed_time)

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