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
        _, a, _ = model.evaluate(self.mesh_vertices, self.mesh_faces, self.body_density, x[0:3])
        a = - np.array(a)
        trajectory_info[len(x):(len(x)+len(a)), 0] = a

        # Initialization
        t = np.empty(it_max + 1, dtype = np.float64)
        t[0] = self.start_time

        rk_eval_matrix = np.empty((len(x),13), dtype = np.float64)
        it = 0

        # Main loop for Runge-kutta 8(7)-13M formulae
        while (t[it] < self.final_time) and (h >= h_min) and it < it_max: 

            if (t[it] + h) > self.final_time:
                h = self.final_time - t[it]

            rk_eval_matrix[:,0] = self.equation_of_motion(h, x)
            for j in range(11):
                rk_eval_matrix[:,j+1] = self.equation_of_motion(h, x + h*rk_eval_matrix.dot(self.a[j+1,:])) #if a(r(t), v(t), t) depends on time, then t=h*self.c[j+1] in this method call. However, with polyhedral model which only depend on r(t), we have a(r)*dt (where dt=h)

            # Two solutions
            solution_low_dim = x + h*rk_eval_matrix.dot(self.b[0,:])
            solution_high_dim = x + h*rk_eval_matrix.dot(self.b[1,:])

            # Truncation error and acceptable error
            truncation_error = np.linalg.norm(solution_low_dim - solution_high_dim, np.inf)
            print("truncation_error: ", truncation_error)
            tau = relative_error_tol * max(np.linalg.norm(x, np.inf), 1)
            print("tau: ", tau)

            # Updating solution if error is acceptable
            if truncation_error <= tau:
                t[it+1] = t[it] + h
                x = solution_high_dim

                # Saving current trajectory information
                trajectory_info[0:len(x), it+1] = x

                print("Option 1")
                _, a, _ = model.evaluate(self.mesh_vertices, self.mesh_faces, self.body_density, x[0:3])
                a = - np.array(a)

                #print("size a: ", a.shape)
                #print("size x: ", x.shape)
                #print("sixe trajectory_info: ", trajectory_info.shape)
                #print("it: ", it)

                trajectory_info[(len(x)):(len(x)+len(a)), it+1] = a
                reject = 0

            else:
                print("Option 2")
                n_reject = n_reject + 1
                reject = 1

            # Adaptive Step size control
            if truncation_error == 0:
                print("Option 3")
                truncation_error = np.finfo(float).eps * 10
                print("New trunc error: ", truncation_error)
            
            h = min(h_max, safety_parameter * h * ((tau/truncation_error)**power))

            if np.abs(h) <= np.finfo(float).eps:
                if reject == 0:
                    print("Warning:  Step size is very small, a small-step correction has been made.")
                    h = np.finfo(float).eps * 100
                    x[0:2] = x[0:3]
                else:
                    print("Error:  Step is too small.")
                    x[0:2] = x[0:3]
            
            # Update iteration counter
            it += 1
            print("it: ", it)
            print("final time: ", self.final_time)
            print("current time: ", t[it])
            #x[0:2] = x[0:3]

        if t[it-1] < self.final_time:
            print("Error:  Final time of integration was never reached.")
        
        elif it == it_max:
            print("Message:  Maximal number of iterations reached at time t=", t[it-1], " [s].")

        return np.vstack((trajectory_info[:, 0:it-1], np.array(t[0:it-1])))