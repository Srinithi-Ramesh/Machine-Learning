Note: The program is written for python 3.6 

1. IMPLEMENT GRADIENT DESCENT
    2. 
       
       Effect of initial guess:
            The initial guess has negligible effect on the convergence of the cost function. This is due to the fact
       that, farther the point is from optimal point, higher will be the gradient which will update the features by a 
       bigger amount. Hence, for a given step size and convergence criteria, initial guess will have very less effect on
       the convergence rate.
            This applies only for convex functions (functions with only one global minimum and no local minimum). For
       non-convex functions, based on the point we start we will end up in the nearest local minimum (if we are lucky, 
       it can be global minimum too!)
       
       Effect of step size:
            It is always better to choose a smaller step size, as higher step size will change the parameters by very big
       values which will cause the optimization to overshoot and the function will diverge. However, choosing a very
       small step size will update the parameters by very small amount at every iteration, and hence the function will
       take a lot of time(iterations) to converge. Thus, choosing an optimal step size is vital in optimization problems.
       
       Effect of convergence criterion:
            The effect of convergence criteria is pretty straightforward. Smaller the value is more accurate will the 
       model be, at the cost of convergence rate and vice versa.


2. LINEAR BASIS FUNCTION REGRESSION
    3. The effect of initial guess, step size and convergence criterion is similar to that discussed in the previous
       section. In this case, the function that we took is the SSE which represent the sum of squares of error 
       of the model. The curve represented by this function will exponentially reach towards zero. Hence all the weights
       should be closer to 0.
       We took gradient descent and BFGS optimizers and following are the results we obtained:
            Gradient Descent:
                M:          1
                iterations: 306
                Weights:    [0.124, -0.797]
            BFGS:
                M:          1
                iterations: 3
                Weights:    [0.820, -1.267]
       From this equation we can see that BFGS algorithm converges faster towards the optimal solution than the Gradient
       Descent algorithm. Since, BFGS is a quasi-Newton method, it will require fewer iterations to reach the optimal 
       solution 
       
3. RIDGE REGRESSION
    2. 
        Try 1: M = 1, lambda = 0.001
                    Gradient Descent  :                                   35.1055685468
                    Gradient Descent using Finite Differences Theorem  :  34.9291824893
                    Using scipy.fmin_bfgs  :                              35.1055687675
                    With Ridge Regression  :                              35.1015432673
                    With Ridge Regression using Numerical Method  :       35.042836637
        
        Try 2: M = 1, lambda = 0.01
                    Gradient Descent  :                                   35.1055685468
                    Gradient Descent using Finite Differences Theorem  :  34.9291824893
                    Using scipy.fmin_bfgs  :                              35.1055687675
                    With Ridge Regression  :                              35.065402755
                    with Ridge Regression using Numerical Method  :       34.8657016051
                
        Try 3: M = 1, lambda = 1
                    Gradient Descent  :                                   35.1055685468
                    Gradient Descent using Finite Differences Theorem  :  34.9291824893
                    Using scipy.fmin_bfgs  :                              35.1055687675
                    With Ridge Regression  :                              31.9156274762
                    with Ridge Regression using Numerical Method  :       28.1313204639

        Try 4: M = 2, lamba = 0.01
                    Gradient Descent  :                                   32.3373250391
                    Gradient Descent using Finite Differences Theorem  :  32.4928527818
                    Using scipy.fmin_bfgs  :                              32.3373253499
                    With Ridge Regression  :                              39.9367921302
                    with Ridge Regression using Numerical Method  :       6.08977603328e+32
 
        Try 5: M = 3, lambda = 0.001
                    Gradient Descent  :                                   27.2314728086
                    Gradient Descent using Finite Differences Theorem  :  3.73121546145e+30
                    Using scipy.fmin_bfgs  :                              27.2314752235
                    With Ridge Regression  :                              27.2300659254
                    with Ridge Regression using Numerical Method  :       7.35941946194e+34
                    
        Try 6: M = 3, lambda = 0.1
                    Gradient Descent  :                                   27.2314728086
                    Gradient Descent using Finite Differences Theorem  :  3.73121546145e+30
                    Using scipy.fmin_bfgs  :                              27.2314752235
                    With Ridge Regression  :                              27.1473697373
                    with Ridge Regression using Numerical Method  :       8.37544494283e+34
                    
        Try 7: M = 4, lambda = 0.1
                    Gradient Descent  :                                   56.7208005775
                    Gradient Descent using Finite Differences Theorem  :  7.00931937569e+32
                    Using scipy.fmin_bfgs  :                              56.7206258463
                    With Ridge Regression  :                              45.9699487961
                    with Ridge Regression using Numerical Method  :       1.95897908419e+36
        
        From the above observations the best model for the given dataset is:
            Linear Regression with Ridge Regularization with M = 3 and lambda = 0.1
            Weights =  [  7.37837775e-01   1.59479508e+00   6.61547813e-02  -2.50032679e-01 ]