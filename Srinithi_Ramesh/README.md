Note: The program is written for python 3.6 

1. IMPLEMENT GRADIENT DESCENT
    2. Effect of initial guess:
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
       Descent algorithm.
       
3. RIDGE REGRESSION
    2. 