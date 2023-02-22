
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
#             Runge-Kutta-Fehlberg 45                #
######################################################

def rkf(self,a,b,y0,tol,hmax,hmin,inHigher,inLower):
    """
    rkf(f,a,b,y0,tol,hmax,hmin,inHigher,inLower) -> [array(),array()]
    
    Runge-Kutta-Fehlberg which combines RK 4th and RK 5h
    order for solving an IVP in the form:
            / y'(x) = f(x,y)
            \ y(a) = y0
    
    RK5 for truncation error:
            y5_i+1 = y5_i + d1*k1 + d3*k3 + d4*k4 + d5*k5 + d6*k6
            
    RK4 for local error estimation:
            y4_i+1 = y4_i + c1*k1 + c3*k3 + c4*k4 + c5*k5
            
    Evaluations:
            k1 = h * f( x, y )
            k2 = h * f( x + a2 * h , y + b21 * k1)
            k3 = h * f( x + a3 * h , y + b31 * k1 + b32 * k2)
            k4 = h * f( x + a4 * h , y + b41 * k1 + b42 * k2 + b43 * k3 )
            k5 = h * f( x + a5 * h , y + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4 )
            k6 = h * f( x + a6 * h , y + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5 )
    
    Params
    ----------
        f        -> (function) to be solved
        a        -> (float) initial interval value
        b        -> (float) end interval value
        y0       -> (float) value of the function at point a
        tol      -> (float) tolerance for truncation error
        hmax     -> (float) max step size
        hmin     -> (float) min step size
        inHigher -> (float) higher increment for step variation
        inLower  -> (float) lower increment for step variation
    Return
    -------
        X     -> (Array NumPy) independent variable
        Y     -> (Array NumPy) solution value of the function
        P     -> (Array NumPy) of steps used according to X
    """

    # Coefficients related to the independent variable of the evaluations
    a2  =   2.500000000000000e-01  #  1/4
    a3  =   3.750000000000000e-01  #  3/8
    a4  =   9.230769230769231e-01  #  12/13
    a5  =   1.000000000000000e+00  #  1
    a6  =   5.000000000000000e-01  #  1/2

    # Coefficients related to the dependent variable of the evaluations
    b21 =   2.500000000000000e-01  #  1/4
    b31 =   9.375000000000000e-02  #  3/32
    b32 =   2.812500000000000e-01  #  9/32
    b41 =   8.793809740555303e-01  #  1932/2197
    b42 =  -3.277196176604461e+00  # -7200/2197
    b43 =   3.320892125625853e+00  #  7296/2197
    b51 =   2.032407407407407e+00  #  439/216
    b52 =  -8.000000000000000e+00  # -8
    b53 =   7.173489278752436e+00  #  3680/513
    b54 =  -2.058966861598441e-01  # -845/4104
    b61 =  -2.962962962962963e-01  # -8/27
    b62 =   2.000000000000000e+00  #  2
    b63 =  -1.381676413255361e+00  # -3544/2565
    b64 =   4.529727095516569e-01  #  1859/4104
    b65 =  -2.750000000000000e-01  # -11/40

    # Coefficients related to the truncation error
    # Obtained through the difference of the 5th and 4th order RK methods:
    #     R = (1/h)|y5_i+1 - y4_i+1|
    r1  =   2.777777777777778e-03  #  1/360
    r3  =  -2.994152046783626e-02  # -128/4275
    r4  =  -2.919989367357789e-02  # -2197/75240
    r5  =   2.000000000000000e-02  #  1/50
    r6  =   3.636363636363636e-02  #  2/55

    # Coefficients related to RK 4th order method
    c1  =   1.157407407407407e-01  #  25/216
    c3  =   5.489278752436647e-01  #  1408/2565
    c4  =   5.353313840155945e-01  #  2197/4104
    c5  =  -2.000000000000000e-01  # -1/5

    # Init x and y with initial values a and y0
    # Init step h with hmax, taking the biggest step possible
    x = a
    y = y0
    h = hmax

    # Init vectors to be returned
    xx = np.array( [x] )
    yy = np.array([y])
    pp = np.array( [h] )

    while x < b - h:

        # Store evaluation values
        k1 = h * self.equation_of_motion( x, y )
        k2 = h * self.equation_of_motion( x + a2 * h, y + b21 * k1 )
        k3 = h * self.equation_of_motion( x + a3 * h, y + b31 * k1 + b32 * k2 )
        k4 = h * self.equation_of_motion( x + a4 * h, y + b41 * k1 + b42 * k2 + b43 * k3 )
        k5 = h * self.equation_of_motion( x + a5 * h, y + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4 )
        k6 = h * self.equation_of_motion( x + a6 * h, y + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5 )

        # Calulate local truncation error
        r = np.linalg.norm( r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6 ) / h
        # If it is less than the tolerance, the step is accepted and RK4 value is stored
        if r <= tol:
            x = x + h
            y = y + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
            xx = np.append( xx, x )
            yy = np.append( yy, y)
            pp = np.append( pp, h )

        # Prevent zero division
        if r == 0: r = pp[-1]
        
        # Calculate next step size
        h = h * min( max( 0.84 * ( tol / r )**0.25, inLower ), inHigher )

        # Upper limit with hmax and lower with hmin
        h = hmax if h > hmax else hmin if h < hmin else h

        print("Step size: ", h)
        print("Current ime: ", x, " [s].")
    
    trajectory_info = np.empty((6, int(len(yy)/6)), dtype=np.float64)
    yy = np.array(yy)
    yy_split = np.array_split(yy, int(len(yy)/6))
    for i in range(0, int(len(yy)/6)):
        trajectory_info[:,i] = yy_split[i]

    return xx,np.array(yy_split),pp








######################################################
#              Runge-Kutta-Fehlberg 78               #
######################################################

def butcher_table_rkf78(self):
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

    t0 = self.start_time
    tf = self.final_time
    h0 = self.time_step
    n_max = 10000
    tol_min = 1e-12
    tol_max = 1e-8
    h_min = 1e-6
    h_max = 10

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
    


        #k_matrix = np.zeros((len(x),13), dtype=np.float64)
        #for i in range(13):
        #    if i == 0:
        #        k_value = 0
        #    else:
        #        for j in range(i):
        #            k_value += self.a[i,j] * k_matrix[:,j]
        #    k_matrix[:,i] = self.equation_of_motion(t, x + h*k_value)
        #
        #y = x + h*np.matmul(k_matrix, self.c)
        #y_hat = x + h*np.matmul(k_matrix, self.c_hat)


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



def alg_rkf78(self,x):

    it_max = 10000

    # Storing trajectory info
    t = np.empty(it_max+1, dtype = np.float64)
    t[0] = self.start_time

    trajectory_info = np.empty((6,it_max+1), dtype = np.float64)
    trajectory_info[0:len(x), 0] = x

    it = 0
    h = self.time_step

    while (t[it] < self.final_time) and it < it_max:
        x_it, h_next, t = self.rkf78(x, h, t[it])
        h = h_next
        trajectory_info[0:len(x_it), it+1] = x_it

        print("Final time: ", self.final_time, " [s].")
        print("Current time: ", t, " [s].")
        print("Iteration: ", it)

        it += 1

    return trajectory_info


def rkf78(self, x, h, t):

    # Tolerance 
    tol = 1e-12
    error_tol = tol/(self.final_time - t)
    error_factor = -41/840

    # Scaling factors
    k_min = 0.125
    k_max = 4.0
    power = 1/7.0

    # Initialization
    it = 0
    it_max = 2000
    last_it = 0
    x_it = np.array(x)

    if (self.final_time < t) or (h <= 0):
        x_new = x_it
        h_next = h
        return x_new, h_next, t
    
    h_next = h
    x_new = x_it

    if self.final_time == t:
        x_new = x_it
        h_next = h
        return x_new, h_next, t

    if (h > (self.final_time-t)):
        h = self.final_time - t
        last_it = 1

    # Main loop
    while (t < self.final_time) and (it < it_max):
        scale = 1.0
        for i in range(11):
            rk_eval_matrix = np.empty((len(x_it),12), dtype = np.float64)
            for j in range(11):
                rk_eval_matrix[:,j] = self.equation_of_motion(h, x_it + h*rk_eval_matrix.dot(self.a[j,:])) #if a(r(t), v(t), t) depends on time, then t=h*self.c[j+1] in this method call. However, with polyhedral model which only depend on r(t), we have a(r)*dt (where dt=h)

            k12 =  self.equation_of_motion(h, x_it + h * (self.a[12, 0]*rk_eval_matrix[:,0]
                + self.a[12,3]*rk_eval_matrix[:,3] + self.a[12,4]*rk_eval_matrix[:,4] + self.a[12,5]*rk_eval_matrix[:,5]
                + self.a[12,6]*rk_eval_matrix[:,6] + self.a[12,7]*rk_eval_matrix[:,7]
                + self.a[12,8]*rk_eval_matrix[:,8] + self.a[12,9]*rk_eval_matrix[:,9] + rk_eval_matrix[:,11]))

            #print("Shape rk_eval: ", rk_eval_matrix.shape)
            #print("Shape k12: ", k12.shape)
            #print("x: ", x.shape)
            #print("c: ", len(self.c))


            #rk_eval_matrix = np.concatenate((rk_eval_matrix, k12), axis = 1)

            #print("Shape rk_eval: ", rk_eval_matrix.shape)

            x_new = x_it + h * (rk_eval_matrix.dot(self.c[0:len(self.c)-1]) + k12*self.c[-1])
            rk_error = error_factor * (rk_eval_matrix[:,0] + rk_eval_matrix[:,10] - rk_eval_matrix[:,11] - k12)
            trunc_error = np.linalg.norm(rk_error)

            if trunc_error == 0:
                scale = k_max
                continue

            if np.linalg.norm(x_new) == 0:
                xx = error_tol
            else:
                xx = np.linalg.norm(x_new)
            
            scale = 0.8 * (((error_tol*xx)/trunc_error)**power)
            scale = min(max(scale, k_min), k_max)

            if trunc_error < (error_tol*xx):
                continue

            h = h * scale

            if (t + h) > self.final_time:
                h = self.final_time - t[it]
            elif (t + h + 0.5*h > self.final_time):
                h = 0.5 * h

        print("12 attemps finished")

        if i >= 11:
            h_next = h * scale
            return x_new, h_next, t
        
        # Updating step:
        x_it = x_new
        t = t + h
        h = h * scale
        h_next = h


        print("internal t: ", t)
        print("internal h: ", h)

        if last_it == 1:
            continue

        if (t + h) > self.final_time:
            last_it = 1
            h = self.final_time - t[it]
        elif (t + h + 0.5*h) > self.final_time:
            h = 0.5*h

        it = it + 1

    print("Finished a full internal cycle")
    return x_new, h_next, t





######################################################
#              Dormand-Prince 8(7)-13M               #
######################################################


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
    
    a = np.array([a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11])
    b = np.array([b7, b8])
    c = np.array(c)
    return a, b, c



def DP_8713M(self, x):

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



