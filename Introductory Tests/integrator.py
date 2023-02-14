
import numpy as np

# For computing acceleration and potential
import polyhedral_gravity as model




class integrator:
    """ 
    Sets up the user defined numerical integrator algorithm of choice.
    The object holds attributes in terms of variables and constants that
    are used for trajectory propagation. 
    The methods of the class mainly defines numerical integrators used 
    for the trajectory propagation.

    Dependencies:
        - Numpy
        - polyhedral-gravity-model
        - PyVista
    """

    def __init__(self, body_mesh, mesh_vertices, mesh_faces, body_density, target_altitude, final_time, start_time, time_step, algorithm):
        """
        Defines all attributes connected to the numerical integrators of this class. 

        Args:
            body_density (float):    Mass density of body of interest
            target_altitude (float): Target altitude for satellite trajectory. 
            final_time (int):        Final time for integration.
            start_time (int):        Start time for integration of trajectory (often zero)
            time_step (int):         Step size for integration. 
            lower_bounds (float):    Lower bounds for domain of initial state.
            upper_bounds (float):    Upper bounds for domain of initial state. 
            algorithm (str):         User defined algorithm of c
         """

        # Attributes for computinga acceleration
        self.body_mesh = body_mesh
        self.mesh_vertices = mesh_vertices
        self.mesh_faces = mesh_faces

        # Additional hyperparameters
        self.body_density = body_density     
        self.target_altitude = target_altitude     
        self.final_time = final_time      
        self.start_time = start_time                
        self.time_step = time_step

        # Required algorithm parameters
        self.algorithm = algorithm
        if self.algorithm == "RKF78":
            self.b, self.a, self.c, self.c_hat = self.butcher_table_rkf78()

        elif self.algorithm == "DP8713M":
            self.a, self.b, self.c = self.butcher_table_dp8713()


    def run_integration(self, x):
        """ Calls the correct numerical integration algorithm.

        Args:
            x: State vector containing position and velocity expressed in three dimensions.

        Returns:
            trajectory_info: Numpy array containing information on position and velocity at every time step.
        """

        if self.algorithm == "Euler":
            trajectory_info = self.euler_approx(x)

        elif self.algorithm == "RKF78":
            trajectory_info = self.new_rkf78(x)
        
        elif self.algorithm == "DP8713M":
            trajectory_info = self.DP_8713M(x)

        return trajectory_info
        


    # Used by all RK-type algorithms
    def equation_of_motion(self, _, x):
        """ State update equation for RK-type algorithms. 

        Args:
            _ : Time value (not needed as of now)
            x : State vector containing position and velocity expressed in three dimensions.

        Returns:
            State vector used for computing state at the following time step.
        """
        _, a, _ = model.evaluate(self.mesh_vertices, self.mesh_faces, self.body_density, x[0:3])
        a = - np.array(a)
        kx = x[3:6] 
        kv = a 
        return np.concatenate((kx, kv))



    ######################################################
    #        Semi-Implicit Euler approximation           #
    ######################################################

    def euler_approx(self, x):
        """euler_approx uses euler's method as numerical integrator for approximating the trajectory. 

        Args:
            x: State vector containing information on position and velocity in 3 dimensions.
            time_list: List of all the times corresponding to a state.
            Trajectory_info: Array containing values on state, acceleration and time at each time step for the trajectory.

        Returns:
            Complete trajectory information stored in trajectory_info.
        """
            # Array containing times for summation
        time_list = np.arange(self.start_time, self.final_time, self.time_step)

        trajectory_info = np.empty(9, len(time_list))
        trajectory_info[0:6, 0] = np.transpose(x)
        
        # Initial information
        r = np.transpose(x[0:3]) # Start Position
        v = np.transpose(x[3:6]) # Initial velocity

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
            trajectory_info[0:3,i] = r
            trajectory_info[3:6,i] = v
            trajectory_info[6:9, i-1] = a
            i += 1

        # Adding final acceleration to memory 
        _, a, _ = model.evaluate(self.mesh_vertices, self.mesh_faces, self.body_density, r)
        a = - np.array(a)
        trajectory_info[6:9, i-1] = a

        return np.vstack(trajectory_info, time_list)







    ######################################################
    #              Runge-Kutta-Fehlberg 78               #
    ######################################################

    def butcher_table_rkf78(self):
        """ Butcher table/tablau for Runge-Kutta-Fehlberg 7(8) method.

        Returns:
            All table values defined for the RKF-78 method. 
        """
        a0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
        a1 = [2/27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        a2 = [1/36, 1/12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        a3 = [1/24, 0, 1/8, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        a4 = [5/12, 0, -25/16, 25/16, 0, 0, 0, 0, 0, 0, 0, 0]
        a5 = [1/20, 0, 0, 1/4, 1/5, 0, 0, 0, 0, 0, 0, 0]
        a6 = [-25/108, 0, 0, 125/108, -65/27, 125/54, 0, 0, 0, 0, 0, 0]
        a7 = [31/300, 0, 0, 0, 61/225, -2/9, 13/900, 0, 0, 0, 0, 0]
        a8 = [2, 0, 0, -53/6, 704/45, -107/9, 67/90, 3, 0, 0, 0, 0]
        a9 = [-91/108, 0, 0, 23/108, -976/135, 311/54, -19/60, 17/6, -1/12, 0, 0, 0]
        a10 = [2383/4100, 0, 0, -341/164, 4496/1025, -301/82, 2133/4100, 45/82, 45/164, 18/41, 0, 0]
        a11 = [37/205, 0, 0, 0, 0, -6/41, -3/205, -3/41, 3/41, 6/41, 0, 0]
        a12 = [-1777/4100, 0, 0, -341/164, 4496/1025, -289/82, 2193/4100, 51/82, 33/164, 12/41, 0, 1]

        b = [0, 2/27, 1/9, 1/6, 5/12, 1/2, 5/6, 1/6, 2/3, 1/3, 1, 0, 1]

        c = [41/840, 0, 0, 0, 0, 34/105, 9/35, 9/35, 9/280, 9/280, 41/840, 0, 0]
        c_hat = [0, 0, 0, 0, 0, 34/105, 9/35, 9/35, 9/280, 9/280, 0, 41/840, 41/840]

        a = np.array([a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12])
        b = np.array(b)
        c = np.array(c)
        c_hat = np.array(c_hat)

        return b, a, c, c_hat
        


    def new_rkf78(self,x):
        """
        Integrates Newton's equations of motion within a user defined time interval
        using Runge-Kutta-Fehlberg 7(8) method.

        Returns:
            Complete trajectory information stored in trajectory_info.
        """

        t0 = self.start_time
        tf = self.final_time
        h0 = self.time_step
        n_max = 10000
        tol_min = 1e-12
        tol_max = 1e-12
        h_min = 1e-6
        h_max = 572

        h = h0
        t = t0
        k = 0

        trajectory_info = np.empty((7, n_max), dtype=np.float64)
        trajectory_info[0:6, k] = x
        trajectory_info[6, k] = t

        while (k < n_max) and (t < tf):
            
            if h < h_min:
                h = h_min
            elif h > h_max:
                h = h_max
            
            k0 = self.equation_of_motion(t, x)
            k1 = self.equation_of_motion(t, x + h*(self.a[1,0]*k0))
            k2 = self.equation_of_motion(t, x + h*(self.a[2,0]*k0 + self.a[2,1]*k1))
            k3 = self.equation_of_motion(t, x + h*(self.a[3,0]*k0 + self.a[3,1]*k1 + self.a[3,2]*k2))
            k4 = self.equation_of_motion(t, x + h*(self.a[4,0]*k0 + self.a[4,1]*k1 + self.a[4,2]*k2 + self.a[4,3]*k3))
            k5 = self.equation_of_motion(t, x + h*(self.a[5,0]*k0 + self.a[5,1]*k1 + self.a[5,2]*k2 + self.a[5,3]*k3 + self.a[5,4]*k4))
            k6 = self.equation_of_motion(t, x + h*(self.a[6,0]*k0 + self.a[6,1]*k1 + self.a[6,2]*k2 + self.a[6,3]*k3 + self.a[6,4]*k4 + self.a[6,5]*k5))
            k7 = self.equation_of_motion(t, x + h*(self.a[7,0]*k0 + self.a[7,1]*k1 + self.a[7,2]*k2 + self.a[7,3]*k3 + self.a[7,4]*k4 + self.a[7,5]*k5 + self.a[7,6]*k6))
            k8 = self.equation_of_motion(t, x + h*(self.a[8,0]*k0 + self.a[8,1]*k1 + self.a[8,2]*k2 + self.a[8,3]*k3 + self.a[8,4]*k4 + self.a[8,5]*k5 + self.a[8,6]*k6 + self.a[8,7]*k7))
            k9 = self.equation_of_motion(t, x + h*(self.a[9,0]*k0 + self.a[9,1]*k1 + self.a[9,2]*k2 + self.a[9,3]*k3 + self.a[9,4]*k4 + self.a[9,5]*k5 + self.a[9,6]*k6 + self.a[9,7]*k7 + self.a[9,8]*k8))
            k10 = self.equation_of_motion(t, x + h*(self.a[10,0]*k0 + self.a[10,1]*k1 + self.a[10,2]*k2 + self.a[10,3]*k3 + self.a[10,4]*k4 + self.a[10,5]*k5 + self.a[10,6]*k6 + self.a[10,7]*k7 + self.a[10,8]*k8 + self.a[10,9]*k9))
            k11 = self.equation_of_motion(t, x + h*(self.a[11,0]*k0 + self.a[11,1]*k1 + self.a[11,2]*k2 + self.a[11,3]*k3 + self.a[11,4]*k4 + self.a[11,5]*k5 + self.a[11,6]*k6 + self.a[11,7]*k7 + self.a[11,8]*k8 + self.a[11,9]*k9 + self.a[11,10]*k10))
            k12 = self.equation_of_motion(t, x + h*(self.a[12,0]*k0 + self.a[12,1]*k1 + self.a[12,2]*k2 + self.a[12,3]*k3 + self.a[12,4]*k4 + self.a[12,5]*k5 + self.a[12,6]*k6 + self.a[12,7]*k7 + self.a[12,8]*k8 + self.a[12,9]*k9 + self.a[12,10]*k10 + self.a[12,11]*k11))
        
            y = x + h*(self.c[0]*k0 + self.c[1]*k1 + self.c[2]*k2 + self.c[3]*k3 + self.c[4]*k4 + self.c[5]*k5 + self.c[6]*k6 + self.c[7]*k7 + self.c[8]*k8 + self.c[9]*k9 + self.c[10]*k10)
            y_hat = x + h*(self.c_hat[0]*k0 + self.c_hat[1]*k1 + self.c_hat[2]*k2 + self.c_hat[3]*k3 + self.c_hat[4]*k4 + self.c_hat[5]*k5 + self.c_hat[6]*k6 + self.c_hat[7]*k7 + self.c_hat[8]*k8 + self.c_hat[9]*k9 + self.c_hat[10]*k10 + self.c_hat[11]*k11 + self.c_hat[12]*k12)

            trunc_error = np.linalg.norm(y-y_hat)

            if (trunc_error > tol_max) and (h > h_max):
                h = h/2
            
            else:
                k = k + 1
                t = t + h
                x = y_hat

                trajectory_info[0:6, k] = x
                trajectory_info[6, k] = t
                
                if trunc_error < tol_min:
                    h = 2*h

            print("Succesfull steps: ", k)
            print("Current time: ", t)
            print("Current step size: ", h)
        
        trajectory_info = np.array(trajectory_info)
        return trajectory_info[:, 0:k]








    ######################################################
    #              Dormand-Prince 8(7)-13M               #
    ######################################################


    def butcher_table_dp8713():
        """ Butcher table/tablau for Dormand-Prince 8(7)-13M method.

        Returns:
            All table values defined for the DP-8713M method. 
        """

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
        
        a = np.array([a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11])
        b = np.array([b7, b8])
        c = np.array(c)
        return a, b, c



    def DP_8713M(self, x):
        """
        Integrates Newton's equations of motion within a user defined time interval
        using Dormand-Prince 8(7)-13M method.

        Returns:
            Complete trajectory information stored in trajectory_info.
        """

        # Relative error tolerance for solution vector
        relative_error_tol = 1e-1

        # Step rejection control parameters
        safety_factor = 0.9
        alpha = 0.7/8  # alpha=0.7/p, where p=order of the runge-kutta formulae

        # Maximum number of iterations
        it_max = 15000 

        # Numpy Arrays to store trajectory information
        trajectory_info = np.empty((9,it_max+1), dtype = np.float64)
        trajectory_info[0:len(x), 0] = x
        _, a, _ = model.evaluate(self.mesh_vertices, self.mesh_faces, self.body_density, x[0:3])
        a = - np.array(a)
        trajectory_info[len(x):(len(x)+len(a)), 0] = a

        # Initialization
        t = np.empty(it_max+1, dtype = np.float64)
        t[0] = self.start_time

        rk_eval_matrix = np.empty((len(x),13), dtype = np.float64)
        it = 0

        h = 10 

        # Main loop for Runge-kutta 8(7)-13M formulae
        while (t[it] < self.final_time) and it < it_max: 

            rk_eval_matrix[:,0] = h * self.equation_of_motion(h, x)
            for j in range(11):
                # Embedded solution
                if j+1 == 11:
                    solution_low_dim = x + h*rk_eval_matrix.dot(self.b[0,:])
                elif j+1 == 12:
                    rk_eval_matrix[:,j+1] = h * self.equation_of_motion(h, solution_low_dim)
                else:
                    rk_eval_matrix[:,j+1] = h * self.equation_of_motion(h, x + h*rk_eval_matrix.dot(self.a[j+1,:])) #if a(r(t), v(t), t) depends on time, then t=h*self.c[j+1] in this method call. However, with polyhedral model which only depend on r(t), we have a(r)*dt (where dt=h)
                    
            # Two solutions
            #solution_low_dim = x + h*rk_eval_matrix.dot(self.b[0,:])
            solution_high_dim = x + h*rk_eval_matrix.dot(self.b[1,:])
            
            # Truncation error and acceptable error
            truncation_error = np.linalg.norm(solution_low_dim - solution_high_dim, np.inf)

            # Adaptive Step size control parameters
            k_max = 5.0 # new step can be a max of two times as big
            k_min = 0.2 # new step can be min a half as big

            if truncation_error == 0:
                h_new = k_max * h
            
            else:
                h_new = h * min(k_max, max(k_min, safety_factor * ((relative_error_tol/truncation_error)**alpha)))

            if truncation_error > relative_error_tol:
                # Reject current step if the actual error exceeds the tolerance
                h = h_new
                continue

            h = min(self.final_time - t[it], h_new)
            

            # Updating solution if error is acceptable
            t[it+1] = t[it] + h
            x = solution_low_dim

            # Saving current trajectory information
            trajectory_info[0:len(x), it+1] = x
            _, a, _ = model.evaluate(self.mesh_vertices, self.mesh_faces, self.body_density, x[0:3])
            a = - np.array(a)
            trajectory_info[(len(x)):(len(x)+len(a)), it+1] = a

            if self.final_time - t[it] <= 1e-8:
                return np.vstack((trajectory_info[:, 0:it+1], np.array(t[0:it+1])))

            # Update iteration counter
            it += 1

            print("it (# steps): ", it)
            print("final time: ", self.final_time)
            print("current time: ", t[it])

        if t[it] < self.final_time:
            print("Error:  Final time of integration was never reached. We reached t=: ", t[it], " [s].")
        
        elif it == it_max:
            print("Message:  Maximal number of iterations reached at time t=", t[it], " [s].")

        return np.vstack((trajectory_info[:, 0:it], np.array(t[0:it])))



