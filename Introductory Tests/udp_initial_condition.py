# General
import numpy as np

# For computing acceleration and potential
import polyhedral_gravity as model

# For working with the mesh
import mesh_utility

# For Plotting
import pyvista as pv


class udp_initial_condition:
    """ _summary_
    Sets up the user defined problem (udp) for use with pygmo.
    The object holds attributes in terms of variables and constants that
    are used for trajectory propagation. 
    The methods of the class defines the objective function for the optimization problem,
    boundaries for input variables, trajectoru propagation and plotting of results. 

    Dependencies:
        - Numpy
        - polyhedral-gravity-model
        - PyVista
    """

    def __init__(self, body_density, target_altitude, final_time, start_time, time_step, lower_bounds, upper_bounds):
        """__init__ _summary_
        Setup udp attributes.

        Args:
            body_density (float):    Mass density of body of interest
            target_altitude (float): Target altitude for satellite trajectory. 
            final_time (int):        Final time for integration.
            start_time (int):        Start time for integration of trajectory (often zero)
            time_step (int):         Step size for integration. 
            lower_bounds (float):    Lower bounds for domain of initial state.
            upper_bounds (float):    Upper bounds for domain of initial state. 
        """
        # Creating the mesh (TetGen)
        self.body_mesh, self.mesh_vertices, self.mesh_faces = mesh_utility.create_mesh()

        # Additional hyperparameters
        self.body_density = body_density     
        self.target_altitude = target_altitude     
        self.final_time = final_time      
        self.start_time = start_time                
        self.time_step = time_step
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds           
    
    def fitness(self,x):
        """ fitness evaluates the proximity of the satallite to target altitude.

        Args:
            x: State vector containing values for position and velocity of satelite in three dimensions. 

        Returns:
            fitness value (float): Difference between squared values of current and target altitude of satellite.
        """
        fitness_value, _, _, _ = self.compute_trajectory(x)
        return [fitness_value]

    def get_bounds(self):
        """get_bounds returns upper and lower bounds for the domain of the state vector.

        Returns:
            Two one-dimensional arrays for the bounady values of the state vector. 
        """
        return (self.lower_bounds, self.upper_bounds)

    def compute_trajectory(self, x):
        """compute_trajectory computes trajectory of satellite using numerical integation techniques 

        Args:
            x: State vector (position and velocity)

        Returns:
            fintess_values: Evaluation of proximity of satelite to target altitude.
            r_store:        Array containing values on position at each time step for the trajectory.
            v_store:        Array containing values on velocities at each time step for the trajectory.
            a_store:        Array containing values on acceleration at each time step for the trajectory.
        """


        # Fitness value (to be maximized)
        fitness_value = 0
        
        # Numerical integration of Newton's equations of motion (trajectory propagation)
        #r_store, v_store, a_store = self.euler_approx(r, v, time_list, r_store, v_store, a_store)
        trajectory_info = self.dormand_prince_8713M(self, x)

        # Return fitness value for the computed trajectory
        squared_altitudes = trajectory_info[0,:]**2 + trajectory_info[1,:]**2 + trajectory_info[2,:]**2
        fitness_value = np.mean(np.abs(squared_altitudes-self.target_altitude))
        return fitness_value, trajectory_info

    
    def euler_approx(self, r, v, time_list, r_store, v_store, a_store):
        """euler_approx uses euler's method as numerical integrator for approximating the trajectory. 

        Args:
            r: Current position, expressed in thee dimensions.
            time_list: List of all the times corresponding to a state.
            r_store:   Array containing values on position at each time step for the trajectory.
            v_store:   Array containing values on velocity at each time step for the trajectory.
            a_store:   Array containing values on acceleration at each time step for the trajectory.

        Returns:
            Complete trajectory information stored in r_store, v_store and a_store.
        """
        for i in range(1,len(time_list)):
            # Retrieve information at current position
            _, a, _ = model.evaluate(self.mesh_vertices, self.mesh_faces, self.body_density, r)
            a = - np.array(a)

            # Computing velocity and position for next time-step
            v_n = v + self.time_step * a
            r_n = r + self.time_step * v_n

            # Update current velocity and position
            v = v_n
            r = r_n

            # Storing updated trajectory information
            r_store[:,i] = r
            v_store[:,i] = v
            a_store[:,i-1] = a
            i += 1
            
        return r_store, v_store, a_store





    def butcher_table_dp8713(self):

        # Setting up hyperparameters for adaptive stepsize
        a0 = [1/18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        a1 = [1/48, 1/16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        a2 = [1/32, 0, 3/32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        a3 = [5/16, 0, -75/64, 75/64, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        a4 = [3/80, 0, 0, 3/16, 3/20, 0, 0, 0, 0, 0, 0, 0, 0]
        a5 = [29443841/614563906, 0, 0, 77736538/692538347, -28693883/1125000000, 23124283/1800000000, 0, 0, 0, 0, 0, 0, 0]
        a6 = [16016141/946692911, 0, 0, 61564180/158732637, 22789713/633445777, 545815736/2771057229, -180193667/1043307555, 0, 0, 0, 0, 0, 0]
        a7 = [39632708/573591083, 0, 0, -433636366/683701615, -421739975/2616292301, 100302831/723423059, 790204164/839813087, 800635310/3783071287, 0, 0, 0, 0, 0]
        a8 = [246121993/1340847787, 0, 0, -37695042795/15268766246, -309121744/1061227803, -12992083/490766935, 6005943493/2108947869, 393006217/1396673457, 123872331/1001029789, 0, 0, 0, 0]
        a9 = [-1028468189/846180014, 0, 0, 8478235783/508512852, 1311729495/1432422823, -10304129995/1701304382, -48777925059/3047939560, 15336726248/1032824649, -45442868181/3398467696, 3065993473/597172653, 0, 0, 0]
        a10 = [185892177/718116043, 0, 0, -3185094517/667107341, -477755414/1098053517, -703635378/230739211, 5731566787/1027545527, 5232866602/850066563, -4093664535/808688257, 3962137247/1805957418, 65686358/487910083, 0, 0]
        a11 = [403863854/491063109, 0, 0, -5068492393/434740067, -411421997/543043805, 652783627/914296604, 11173962825/925320556, -13158990841/6184727034, 3936647629/1978049680, -160528059/685178525, 248638103/1413531060, 0, 0]

        b7 = [13451932/455176623, 0, 0, 0, 0, -808719846/976000145, 1757004468/5645159321, 656045339/265891186, -3867574721/1518517206, 465885868/322736535, 53011238/667516719, 2/45, 0]
        b8 = [14005451/335480064, 0, 0, 0, 0, -59238493/1068277825, 181606767/758867731, 561292985/797845732, -1041891430/1371343529, 760417239/1151165299, 118820643/751138087, -528747749/2220607170,  1/4]
        
        c = [1/18, 1/12, 1/8, 5/16, 3/8, 59/400, 93/200, 5490023248/9719169821, 13/20, 1201146811/1299019798, 1, 1]
        
        self.a = np.array([a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11])
        self.b = np.array([b7, b8])
        self.c = c
    





    def equation_of_motion(self, dt, x):
        _, a, _ = model.evaluate(self.mesh_vertices, self.mesh_faces, self.body_density, x[0:2])
        a = - np.array(a)
        kx = x[3:5] * dt
        kv = a * dt
        return np.concatenate((kx, kv))
    
    
    def dormand_prince_8713M(self, x):

        # Defining boundaries for step size
        h_min = 16 * np.finfo(float).eps * np.abs(self.start_time)
        h_max = (self.final_time - self.start_time)/2.5

        # Initial step size
        h = self.time_step
        if h > 0.1:
            h = 0.1
        if h > h_max:
            h = h_max

        # Relative error tolerance for solution vector
        relative_error_tol = 1e-6

        # Step rejection control parameters
        n_reject = 0
        reject = 0
        safety_parameter = 0.9
        power = 1/8  # q=1+p=8, where p=order of the runge-kutta formulae

        # Maximum number of iterations
        it_max = int((self.final_time - self.start_time)/h) * 2

        # Numpy Arrays to store trajectory information
        trajectory_info = np.empty((9,it_max), dtype = np.float64)
        trajectory_info[0:len(x), 0] = x
        _, a, _ = model.evaluate(self.mesh_vertices, self.mesh_faces, self.body_density, x[0:2])
        a = - np.array(a)
        trajectory_info[(len(x)+1):(len(x)+1+len(a)), 0] = a

        # Initialization
        t = np.empty(it_max + 1, dtype = np.float64)
        t[0] = self.start_time

        rk_eval_matrix = np.empty((len(x),13), dtype = np.float64)
        it = 0

        # Main loop for Runge-kutta 8(7)-13M formulae
        while (t[it] < self.final_time) and (h >= h_min) and it < it_max: 

            if (t[it] + h) > self.final_time:
                h = self.final_time - t[it]

            if it == 0:
                dt = 0
            else:
                dt = t[it] - t[it-1]

            rk_eval_matrix[:,0] = self.equation_of_motion(dt, x)
            for j in range(11):
                rk_eval_matrix[:,j+1] = self.equation_of_motion(dt + h*self.c[j+1], x + h*rk_eval_matrix*self.a[j+1,:])

            # Two solutions
            solution_low_dim = x + h*rk_eval_matrix*self.b[0,:]
            solution_high_dim = x + h*rk_eval_matrix*self.b[1,:]

            # Truncation error
            truncation_error = np.linalg.norm(solution_low_dim - solution_high_dim)

            # Estimating error and acceptable error
            step_error = np.linalg.norm(truncation_error, np.inf)
            tau = relative_error_tol * max(np.linalg.norm(x, np.inf), 1)

            # Updating solution if error is acceptable
            if step_error <= tau:
                t[it+1] = t[it] + h
                x = solution_high_dim

                # Saving current trajectory information
                trajectory_info[0:len(x), it+1] = x
                _, a, _ = model.evaluate(self.mesh_vertices, self.mesh_faces, self.body_density, x[0:2])
                a = - np.array(a)
                trajectory_info[(len(x)+1):(len(x)+1+len(a)), it+1] = a
                reject = 0

            else:
                n_reject = n_reject + 1
                reject = 1

            # Adaptive Step size control
            if step_error == 0:
                step_error = np.finfo(float).eps * 10
            
            h = min(h_max, safety_parameter * h * ((tau/step_error)**power))

            if np.abs(h) <= np.finfo(float).eps:
                if reject == 0:
                    print("Warning:  Step size is very small, a small-step correction has been made.")
                    h = np.finfo(float).eps * 100
                else:
                    print("Error:  Step is too small.")
            
            # Update iteration counter
            it += 1

        if t[it-1] < self.final_time:
            print("Error:  Final time of integration was never reached.")
        
        elif it == it_max:
            print("Message:  Maximal number of iterations reached at time t=", t[it-1], " [s].")

        return trajectory_info[:, 0:it-1], t[0:it-1]








    def plot_trajectory(self, r_store):
        """plot_trajectory plots the body mesh and satellite trajectory.

        Args:
            r_store: Array containing values on position at each time step for the trajectory.
        """

        # Plotting mesh of asteroid/comet
        mesh_plot = pv.Plotter()
        mesh_plot.add_mesh(self.body_mesh.grid, show_edges=True)
        mesh_plot.show_bounds(grid='front',location='outer',all_edges=True)

        # Plotting trajectory
        trajectory_plot = np.transpose(r_store)
        mesh_plot.add_lines(trajectory_plot, color="red", width=20)

        trajectory_plot = pv.PolyData(np.transpose(r_store[:,-1]))
        mesh_plot.add_mesh(trajectory_plot, color=[1.0, 1.0, 1.0], style='surface')
        
        mesh_plot.show(jupyter_backend = 'panel') 