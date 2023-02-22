

    ######################################################
    #              Dormand-Prince 8(7)-13M               #
    ######################################################


    def butcher_table_dp8713(self) -> Union[np.ndarray, np.ndarray, np.ndarray]:
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



    def old_DP_8713M(self, x: np.ndarray) -> np.ndarray:
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
        a = self.eq_of_motion.compute_acceleration(x[0:3])
        trajectory_info[len(x):(len(x)+len(a)), 0] = a

        # Initialization
        t = np.empty(it_max+1, dtype = np.float64)
        t[0] = self.start_time

        rk_eval_matrix = np.empty((len(x),13), dtype = np.float64)
        it = 0

        h = 10 

        # Main loop for Runge-kutta 8(7)-13M formulae
        while (t[it] < self.final_time) and it < it_max: 

            rk_eval_matrix[:,0] = h * self.eq_of_motion.compute_motion(x, h)
            for j in range(11):
                # Embedded solution
                if j+1 == 11:
                    solution_low_dim = x + h*rk_eval_matrix.dot(self.b[0,:])
                elif j+1 == 12:
                    rk_eval_matrix[:,j+1] = h * self.eq_of_motion.compute_motion(solution_low_dim, h)
                else:
                    #Note: if a(r(t), v(t), t) depends on time, then t=h*self.c[j+1] in this method call. 
                    #      However, with polyhedral model which only depend on r(t), we have a(r)*dt (where dt=h)
                    rk_eval_matrix[:,j+1] = h * self.eq_of_motion.compute_motion(x + h*rk_eval_matrix.dot(self.a[j+1,:]), h) 
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
            a = self.eq_of_motion.compute_acceleration(x[0:3])
            trajectory_info[(len(x)):(len(x)+len(a)), it+1] = a

            if self.final_time - t[it] <= 1e-8:
                return np.vstack((trajectory_info[:, 0:it+1], np.array(t[0:it+1])))

            # Update iteration counter
            it += 1

        if t[it] < self.final_time:
            print("Error:  Final time of integration was never reached. We reached t=: ", t[it], " [s].")
        
        elif it == it_max:
            print("Message:  Maximal number of iterations reached at time t=", t[it], " [s].")

        return np.vstack((trajectory_info[:, 0:it], np.array(t[0:it])))


    def DP_8713M(self, x: np.ndarray) -> np.ndarray:
        """
        Integrates Newton's equations of motion within a user defined time interval
        using Dormand-Prince 8(7)-13M method.

        Returns:
            Complete trajectory information stored in trajectory_info.
        """

        # Relative error tolerance for solution vector
        relative_error_tol = 1e-12

        # Step rejection control parameters
        safety_factor = 0.9
        alpha = 1/8  # alpha=0.7/p, where p=order of the runge-kutta formulae

        # Maximum number of iterations
        it_max = 15000 


        a = np.array(
            [[0.0,                    0.0,                  0.0,      0.0,       0.0,                      0.0,                    0.0,                     0.0,                     0.0,                     0.0,                     0.0,                   0.0,                  0.0, 0.0],
                [1/18,                   1/18,                 0.0,      0.0,       0.0,                      0.0,                    0.0,                     0.0,                     0.0,                     0.0,                     0.0,                   0.0,                  0.0, 0.0],
                [1/12,                   1/48,                 1/16,     0.0,       0.0,                      0.0,                    0.0,                     0.0,                     0.0,                     0.0,                     0.0,                   0.0,                  0.0, 0.0],
                [1/8,                    1/32,                 0.0,      3/32,      0.0,                      0.0,                    0.0,                     0.0,                     0.0,                     0.0,                     0.0,                   0.0,                  0.0, 0.0],
                [5/16,                   5/16,                 0.0,     -75/64,     75/64,                    0.0,                    0.0,                     0.0,                     0.0,                     0.0,                     0.0,                   0.0,                  0.0, 0.0],
                [3/8,                    3/80,                 0.0,      0.0,       3/16,                     3/20,                   0.0,                     0.0,                     0.0,                     0.0,                     0.0,                   0.0,                  0.0, 0.0],
                [59/400,                 29443841/614563906,   0.0,      0.0,       77736538/692538347,      -28693883/1125000000,    23124283/1800000000,     0.0,                     0.0,                     0.0,                     0.0,                   0.0,                  0.0, 0.0],
                [93/200,                 16016141/946692911,   0.0,      0.0,       61564180/158732637,       22789713/633445777,     545815736/2771057229,   -180193667/1043307555,    0.0,                     0.0,                     0.0,                   0.0,                  0.0, 0.0],
                [5490023248/9719169821,  39632708/573591083,   0.0,      0.0,      -433636366/683701615,     -421739975/2616292301,   100302831/723423059,     790204164/839813087,     800635310/3783071287,    0.0,                     0.0,                   0.0,                  0.0, 0.0],
                [13/20,                  246121993/1340847787, 0.0,      0.0,      -37695042795/15268766246, -309121744/1061227803,  -12992083/490766935,      6005943493/2108947869,   393006217/1396673457,    123872331/1001029789,    0.0,                   0.0,                  0.0, 0.0],
                [1201146811/1299019798, -1028468189/846180014, 0.0,      0.0,       8478235783/508512852,     1311729495/1432422823, -10304129995/1701304382, -48777925059/3047939560,  15336726248/1032824649, -45442868181/3398467696,  3065993473/597172653,  0.0,                  0.0, 0.0],
                [1,                      185892177/718116043,  0.0,      0.0,      -3185094517/667107341,    -477755414/1098053517,  -703635378/230739211,     5731566787/1027545527,   5232866602/850066563,   -4093664535/808688257,    3962137247/1805957418, 65686358/487910083,   0.0, 0.0],
                [1,                      403863854/491063109,  0.0,      0.0,      -5068492393/434740067,    -411421997/543043805,    652783627/914296604,     11173962825/925320556,  -13158990841/6184727034,  3936647629/1978049680,  -160528059/685178525,   248638103/1413531060, 0.0, 0.0]], dtype=np.float64
        )

        c = np.array(
            [[0., 13451932/455176623, 0.0, 0.0, 0.0, 0.0, -808719846/976000145, 1757004468/5645159321, 656045339/265891186, -3867574721/1518517206, 465885868/322736535,  53011238/667516719,   2/45,                 0.0],
                [0., 14005451/335480064, 0.0, 0.0, 0.0, 0.0, -59238493/1068277825, 181606767/758867731,   561292985/797845732, -1041891430/1371343529, 760417239/1151165299, 118820643/751138087, -528747749/2220607170, 1/4]], dtype=np.float64
        )

        # Numpy Arrays to store trajectory information
        trajectory_info = np.empty((6,it_max+1), dtype = np.float64)
        trajectory_info[0:6, 0] = x
     

        # Initialization
        t = np.empty(it_max+1, dtype = np.float64)
        t[0] = self.start_time

        rk_eval_matrix = np.zeros((len(x),13), dtype = np.float64)
        it = 0
        h = 800

        # Main loop for Runge-kutta 8(7)-13M formulae
        while (t[it] < self.final_time) and it < it_max: 

            h = min(self.final_time - t[it], h)

            for j in range(12):
                rk_eval_matrix[:,j] = h * self.eq_of_motion.compute_motion(x + rk_eval_matrix.dot(a[j,1:(len(a[0]))]), h)

            solution_low_dim = x + rk_eval_matrix.dot(c[0,1:len(a[0])])
            solution_high_dim = x + rk_eval_matrix.dot(c[1,1:len(a[0])])

            # Truncation error and acceptable error
            truncation_error = np.linalg.norm(solution_low_dim - solution_high_dim, np.inf)
            print("trunc err: ", truncation_error)

            # Adaptive Step size control parameters
            delta = safety_factor * h * ((relative_error_tol/truncation_error)**alpha)

            print("Delta: ", delta)
            print("h: ", h)

            if truncation_error <= relative_error_tol:
                # Updating solution if error is acceptable
                t[it+1] = t[it] + h
                x = solution_low_dim

                # Saving current trajectory information
                trajectory_info[0:6, it+1] = x

                # Update iteration counter
                it += 1

                h = delta * h

            else:
                h = delta * h

            print(t[it])

            if self.final_time - t[it] <= 1e-8:
                return np.vstack((trajectory_info[:, 0:it+1], np.array(t[0:it+1])))

            
        if t[it] < self.final_time:
            print("Error:  Final time of integration was never reached. We reached t=: ", t[it], " [s].")
        
        elif it == it_max:
            print("Message:  Maximal number of iterations reached at time t=", t[it], " [s].")

        return np.vstack((trajectory_info[:, 0:it], np.array(t[0:it])))







    def rkf45(self, x):
        epsilon = 1e-10
        h = 800
        i = 0
        t = self.start_time
        imax = 1000

        info_list = np.empty((7,2000), dtype=np.float64)
        info_list[0:6,0] = x
        info_list[6,0] = t

        while (t < self.final_time) or (i < imax):
            h = min(h, self.final_time-t)
            k1 = h*self.eq_of_motion.compute_motion(x, t)
            k2 = h*self.eq_of_motion.compute_motion(x + (k1/4), t + (h/4))
            k3 = h*self.eq_of_motion.compute_motion(x + (3*k1/32) + (9*k2/32), t + (3*h/8))
            k4 = h*self.eq_of_motion.compute_motion(x + (1932*k1/2197) - (7200*k2/2197) + (7296*k3/2197), t + (12*h/13))
            k5 = h*self.eq_of_motion.compute_motion(x + (439*k1/216) - (8*k2) + (3680*k3/513) - (845*k4/4104), t + h)
            k6 = h*self.eq_of_motion.compute_motion(x - (8*k1/27) + (2*k2) - (3544*k3/2565) + (1859*k4/4104) - (11*k5/40), t + (h/2))
            w1 = x + (25*k1/216) + (1408*k3/2565) + (2197*k4/4104) - (k5/5)
            w2 = x + (16*k1/135) + (6656*k3/12825) + (28561*k4/56430) - (9*k5/50) + (2*k6/55)
            err = np.linalg.norm(w1-w2)/h
            delta = 0.84*(epsilon/err)**(1/4)
            
            if err<=epsilon:
                t = t+h
                x = w1
                i = i+1
                h = delta*h

                info_list[0:6,i] = x
                info_list[6,i] = t

            else:
                h = delta*h

            if t == self.final_time:
                return info_list
            
        return info_list















