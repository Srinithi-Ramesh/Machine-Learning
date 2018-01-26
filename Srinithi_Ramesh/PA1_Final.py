# Implement a basic gradient descent procedure to minimize scalar functions of a vector argument.
# Write it generically, so that you can easily specify the objective function and the function
# to compute the gradient. You should be able to specify the initial guess, the step size and the
# convergence criterion (a threshold such that the algorithm terminates when the objective values
# on two successive steps is below this value).

import math
import matplotlib.pyplot as plt
from collections import Iterable
import numpy as np
import pdb
import random
import pylab as pl
from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_cg

########################################################################################################################
# IMPLEMENT GRADIENT DESCENT
########################################################################################################################

########################################################################################################################
# 1.1
########################################################################################################################


count_values = []
error_values = []


def gradient_descent(f, g, *args, initial_guess, step_size, convergence_criteria):
    count_values.clear()
    error_values.clear()
    error = math.inf
    curr = initial_guess
    count = 0
    while abs(error) > convergence_criteria:
        grad = g(curr, *args)
        if isinstance(curr, Iterable) or isinstance(grad, Iterable):
            updated_values = np.subtract(curr, tuple([step_size * x for x in grad]))
        else:
            updated_values = curr - step_size * grad
        curr_f = f(curr, *args)
        updated_f = f(updated_values, *args)
        error = curr_f - updated_f
        curr = updated_values
        count = count + 1
        count_values.append(count)
        error_values.append(error)
    print("Total iterations: %s" % count)
    return curr


########################################################################################################################
# 1.2
########################################################################################################################


# (x+2y-7)^2 + (2x+y-5)^2
args1 = (1, 2, 7, 5)


def booth_gradient(z, *args):
    x, y = z
    a, b, c, d = args
    df_dx = 2 * a * (a * x + b * y - c) + \
            2 * b * (b * x + a * y - d)
    df_dy = 2 * b * (a * x + b * y - c) + \
            2 * a * (b * x + a * y - d)
    return df_dx, df_dy


def booth(z, *args):
    x, y = z
    a, b, c, d = args
    return (a * x + b * y - c) ** 2 + \
           (b * x + a * y - d) ** 2


def minimize_booth(initial_guess, step_size, convergence_criteria):
    return gradient_descent(booth, booth_gradient, *args1,
                            initial_guess=initial_guess, step_size=step_size,
                            convergence_criteria=convergence_criteria)


# 2+cos(x)+0.5*cos(2*x-0.5)
args2 = (2, 0.5, 2, 0.5)


def nonconvex_gradient(x, *args):
    a, b, c, d = args
    df_dx = - math.sin(x) - b * c * math.sin(c * x - d)
    return df_dx


def nonconvex(x, *args):
    a, b, c, d = args
    return a + math.cos(x) + b * math.cos(c * x - d)


def minimize_nonconvex(initial_guess, step_size, convergence_criteria):
    return gradient_descent(nonconvex, nonconvex_gradient, *args2,
                            initial_guess=initial_guess, step_size=step_size,
                            convergence_criteria=convergence_criteria)


# sin(x + y) + (x - y) ^ 2 - 1.5 * x + 2.5 * y + 1
args3 = (1.5, 2.5, 1)


def mccormick_gradient(z, *args):
    a, b, c = args
    x, y = z
    df_dx = math.cos(x + y) + 2 * (x - y) - a
    df_dy = math.cos(x + y) - 2 * (x - y) + b
    return df_dx, df_dy


def mccormick(z, *args):
    a, b, c = args
    x, y = z
    return math.sin(x + y) + (x - y) ** 2 - a * x + b * y + c


def minimize_mccormick(initial_guess, step_size, convergence_criteria):
    return gradient_descent(mccormick, mccormick_gradient, *args3,
                            initial_guess=initial_guess, step_size=step_size,
                            convergence_criteria=convergence_criteria)


# 0.26(x^2 + y^2) - 0.48xy
args4 = (0.26, 0.48)


def matyas_gradient(z, *args):
    x, y = z
    a, b = args
    df_dx = 2 * a * x - b * y
    df_dy = 2 * a * y - b * x
    return df_dx, df_dy


def matyas(z, *args):
    x, y = z
    a, b, = args
    return (a * ((x ** 2) + (y ** 2))) - (b * x * y)


def minimize_matyas(initial_guess, step_size, convergence_criteria):
    return gradient_descent(matyas, matyas_gradient, *args4,
                            initial_guess=initial_guess, step_size=step_size,
                            convergence_criteria=convergence_criteria)


print("Booth Function (Convex): f(x,y) = (x+2y-7)^2 + (2x+y-5)^2")
print(minimize_booth((2, 2), 0.01, 0.000001))
print("\nNon-convex: f(x) = 2+cos(x)+0.5*cos(2*x-0.5)")
print(minimize_nonconvex(3, 0.01, 0.000001))
print("\nMccormick Function (Non convex): f(x, y) = sin(x + y) + (x - y) ^ 2 - 1.5 * x + 2.5 * y + 1")
print(minimize_mccormick((2, 2), 0.01, 0.000001))
print("\nMatyas Function(Convex): f(x, y) = 0.26(x^2 + y^2) - 0.48xy")
print(minimize_matyas((3, 3), 0.01, 0.000001))

############################################################################################################
# 1.3
############################################################################################################


h = 0.01
error_values_finite = []
count_values_finite = []


def gradient_descent_finite_theorem(f, g, *args, initial_guess, step_size, convergence_criteria):
    error = math.inf
    curr = initial_guess
    count = 0
    while abs(error) > convergence_criteria:
        grad = g(f, curr, *args)
        if isinstance(curr, Iterable) or isinstance(grad, Iterable):
            updated_values = np.subtract(curr, tuple([step_size * x for x in grad]))
        else:
            updated_values = curr - step_size * grad
        curr_f = f(curr, *args)
        updated_f = f(updated_values, *args)
        error = curr_f - updated_f
        curr = updated_values
        count = count + 1
        count_values.append(count)
        error_values.append(error)
    print("Total iterations: %s" % count)
    return curr


def finite_differences_gradient(f, z, *args):
    try:
        length = len(z)
        z = np.asarray(z)
        grad = np.zeros(length)
        for i in range(length):
            z_neg = z.copy()
            z_neg[i] -= 0.5 * h
            z_pos = z.copy()
            z_pos[i] += 0.5 * h
            grad[i] = f(z_pos, *args) - f(z_neg, *args)
        return grad
    except TypeError:
        grad = f(z + h * 0.5, *args) - f(z - h * 0.5, *args)
        return grad


def finite_minimize_booth(initial_guess, step_size, convergence_criteria):
    return gradient_descent_finite_theorem(booth, finite_differences_gradient, *args1,
                                           initial_guess=initial_guess, step_size=step_size,
                                           convergence_criteria=convergence_criteria)


def finite_minimize_nonconvex(initial_guess, step_size, convergence_criteria):
    return gradient_descent_finite_theorem(nonconvex, finite_differences_gradient, *args2,
                                           initial_guess=initial_guess, step_size=step_size,
                                           convergence_criteria=convergence_criteria)


def finite_minimize_mccormick(initial_guess, step_size, convergence_criteria):
    return gradient_descent_finite_theorem(mccormick, finite_differences_gradient, *args3,
                                           initial_guess=initial_guess, step_size=step_size,
                                           convergence_criteria=convergence_criteria)


def finite_minimize_matyas(initial_guess, step_size, convergence_criteria):
    return gradient_descent_finite_theorem(matyas, finite_differences_gradient, *args4,
                                           initial_guess=initial_guess, step_size=step_size,
                                           convergence_criteria=convergence_criteria)


print("\nFinite Differences:")
print("Booth Function (Convex): f(x,y) = (x+2y-7)^2 + (2x+y-5)^2")
print(finite_minimize_booth((2, 2), 0.01, 0.000001))
print("\nNon-convex: f(x) = 2+cos(x)+0.5*cos(2*x-0.5)")
print(finite_minimize_nonconvex(3, 0.01, 0.000001))
print("\nMccormick Function (Non convex): f(x, y) = sin(x + y) + (x - y) ^ 2 - 1.5 * x + 2.5 * y + 1")
print(finite_minimize_mccormick((2, 2), 0.01, 0.000001))
print("\nMatyas Function(Convex): f(x, y) = 0.26(x^2 + y^2) - 0.48xy")
print(finite_minimize_matyas((2, 2), 0.01, 0.000001))

#######################################################################################################################
# 1. 4
#######################################################################################################################

print("\nComparision with scipy optimizers")
print("Booth Function (Convex): f(x,y) = (x+2y-7)^2 + (2x+y-5)^2")
print(minimize_booth((2, 2), 0.01, 0.000001))
booth_min_cg = fmin_cg(f=booth, x0=(2, 2), args=args1)
print(booth_min_cg)

print("\nNon-convex: f(x) = 2+cos(x)+0.5*cos(2*x-0.5)")
print(minimize_nonconvex(3, 0.01, 0.000001))
nonconvex_min_cg = fmin_cg(f=nonconvex, x0=3, args=args2)
print(nonconvex_min_cg)

print("\nMccormick Function (Non convex): f(x, y) = sin(x + y) + (x - y) ^ 2 - 1.5 * x + 2.5 * y + 1")
print(minimize_mccormick((2, 2), 0.01, 0.000001))
mccormick_min_cg = fmin_cg(f=mccormick, x0=(2, 2), args=args3)
print(mccormick_min_cg)

print("\nMatyas Function(Convex): f(x, y) = 0.26(x^2 + y^2) - 0.48xy")
print(minimize_matyas((3, 3), 0.01, 0.000001))
matyas_min_cg = fmin_cg(f=matyas, x0=(2, 2), args=args4)
print(matyas_min_cg)


########################################################################################################################
# LINEAR BASIS FUNCTION REGRESSION
########################################################################################################################

#######################################################################################################################
# 2.1
#######################################################################################################################

def designMatrix(X, order):
    if type(X) is not np.ndarray:
        X = np.asarray(X)
    phi = np.zeros([len(X), order + 1], dtype=np.float)
    phi[:, 0] = np.ones(len(X))
    for i in range(1, order + 1):
        phi[:, i] = (X ** i).flatten()
    return phi


def regressionFit(X, Y, phi):
    return (((np.linalg.inv(phi.T.dot(phi))).dot(phi.T)).dot(Y)).T


# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values
# order is the order of the highest order polynomial in the basis functions
def regressionPlot(X, Y, order):
    phi = designMatrix(X, order)
    args = (Y, phi)
    regression_fit_w = regressionFit(X, Y, phi)
    pl.plot(X.T.tolist()[0], Y.T.tolist()[0], 'gs')
    pts = [[p] for p in pl.linspace(min(X), max(X), 100)]
    Yp_reg_fit = pl.dot(regression_fit_w, designMatrix(pts, order).T)
    pl.plot(pts, Yp_reg_fit.tolist()[0], 'b-')
    pl.show()
    return regression_fit_w


def getData(name):
    data = pl.loadtxt(name)
    X = data[0:1].T
    Y = data[1:2].T
    return X, Y


def bishopCurveData():
    # y = sin(2 pi x) + N(0,0.3),
    return getData('/home/raghavan/Downloads/PA1/curvefitting.txt')


print("\nLinear Basis Function Regression - MLE")
X, Y = bishopCurveData()
print("Optimal weight vector for order 0: \n", regressionPlot(X, Y, 0))
print("Optimal weight vector for order 1: \n", regressionPlot(X, Y, 1))
print("Optimal weight vector for order 3: \n", regressionPlot(X, Y, 3))
print("Optimal weight vector for order 9: \n", regressionPlot(X, Y, 9))


#######################################################################################################################
# 2.2
#######################################################################################################################


def sum_of_squared_errors(w, *args):
    Y, phi = args
    rss = 0
    for i in range(len(Y)):
        rss += (Y[i] - (phi[i, :]).dot(w)) ** 2
    return rss[0]


# Analytical method
def gradient_of_SSE(w, *args):
    Y, phi = args
    return np.subtract(phi.T.dot(phi).dot(w), Y.T.dot(phi))


# Numerical method
def get_gradient_SSE(w, *args):
    df_dw = np.zeros([len(w)])
    for i in range(len(w)):
        w_copy_neg = w.copy()
        w_copy_neg[i] -= 0.5 * h
        w_copy_pos = w.copy()
        w_copy_pos[i] += 0.5 * h
        df_dw[i] = np.subtract(sum_of_squared_errors(w_copy_pos, *args),
                               sum_of_squared_errors(w_copy_neg, *args))
    return df_dw.T


w = np.zeros(2)
phi = designMatrix(X, 1)
args = (Y, phi)
print("\nLinear Basis Function Regression - comparing numerical and analytical gradient for SSE")
print("Analytical gradient at w0=0, w1=0\n", gradient_of_SSE(w, *args))
print("Numerical gradient at w0=0, w1=0\n", get_gradient_SSE(w, *args))
w = regressionFit(X, Y, phi)
w0 = w[:, 0]
w1 = w[:, 1]
print("Analytical gradient at w0=\n", w0, "w1=", w1, gradient_of_SSE(w[0], *args))
print("Numerical gradient at w0=\n", w0, "w1=", w1, get_gradient_SSE(w[0], *args))

#######################################################################################################################
# 2.3
#######################################################################################################################

# Gradient descent on SSE - get_gradient_SSE
w = gradient_descent(sum_of_squared_errors, get_gradient_SSE, *args,
                     initial_guess=np.zeros((2, 1)), step_size=0.1, convergence_criteria=0.000001)

print("\nLinear Basis Function Regression - Gradient Descent")
print("Weights from gradient descent:\n", w[0], "\nIterations: ", len(count_values))

opt_w = fmin_bfgs(sum_of_squared_errors, np.zeros((2, 1)), args=args)

print("Weights from optimal library:\n", opt_w)


#######################################################################################################################
# 3.1
#######################################################################################################################

def sumOfSquaredErrorsRegularized(w, l, *args):
    Y, phi = args
    rss = 0
    for i in range(len(Y)):
        rss += (Y[i] - (phi[i, :]).dot(w.T)) ** 2 + (l * 0.5) * w.dot(w.T)
    return rss[0]


def analyticalRidgeRegression(l, *args):
    Y, phi = args
    w = np.linalg.inv(l * (np.identity(phi.shape[1])) + phi.T.dot(phi)).dot(phi.T).dot(Y)
    w = w.T
    return w


def get_ridge_gradient(w, l, *args):
    df_dw = np.zeros([len(w)])
    for i in range(len(w)):
        w_copy_neg = w.copy()
        w_copy_neg[i] -= 0.5 * 0.1
        w_copy_pos = w.copy()
        w_copy_pos[i] += 0.5 * 0.1
        df_dw[i] = sumOfSquaredErrorsRegularized(w_copy_pos.T, l, *args) - \
                   sumOfSquaredErrorsRegularized(w_copy_neg.T, l, *args)
    return df_dw


def numericalRidgeregression(l, *args, initial_guess, step_size, convergence_criteria):
    error = math.inf
    count = 0
    Y, phi = args
    while abs(error) > convergence_criteria:
        grad = get_ridge_gradient(initial_guess, l, *args)
        updated_w = np.subtract(initial_guess, step_size * grad)
        curr_f = sumOfSquaredErrorsRegularized(initial_guess.T, l, *args)
        updated_f = sumOfSquaredErrorsRegularized(updated_w.T, l, *args)
        error = curr_f - updated_f
        initial_guess = updated_w
        count = count + 1
        if count % 100 == 0:
            print("count", count, "error:", error)
    print("Total iterations: %s \nError: %s" % (count, error))
    return initial_guess


l = 0.001

analytical_regularized_w = analyticalRidgeRegression(l, *args)
numerical_regularized_w = numericalRidgeregression(l, *args, initial_guess=np.zeros(2),
                                                   step_size=0.1, convergence_criteria=0.000001)
print("\nRidge Regularization")
print("Ridge Regularization analytically:\n", analytical_regularized_w)
print("Ridge Regression numerically:\n", numerical_regularized_w)


########################################################################################################################
# 3.2
########################################################################################################################


def regressAData():
    return getData('/home/raghavan/Downloads/PA1/regressA_train.txt')


def regressBData():
    return getData('/home/raghavan/Downloads/PA1/regressB_train.txt')


def validateData():
    return getData('/home/raghavan/Downloads/PA1/regress_validate.txt')


def model_selection(X, Y, order, initial_guess=None):
    phi = designMatrix(X, order)
    args = (Y, phi)
    step_size = 0.1
    convergence_criteria = 0.00001
    if initial_guess is None:
        initial_guess = np.ones((5, order + 1))

    regression_fit_w = regressionFit(X, Y, phi)
    finite_diff_w = gradient_descent_finite_theorem(sum_of_squared_errors, finite_differences_gradient, *args,
                                                    initial_guess=initial_guess[1], step_size=step_size,
                                                    convergence_criteria=convergence_criteria)
    optimized_function_w = fmin_bfgs(sum_of_squared_errors, initial_guess[2], args=args)
    analytical_regularized_w = analyticalRidgeRegression(l, *args)
    numerical_regularized_w = numericalRidgeregression(l, *args, initial_guess=initial_guess[4], step_size=step_size,
                                                       convergence_criteria=convergence_criteria)

    # produce a plot of the values of the function

    pl.plot(X.T.tolist()[0], Y.T.tolist()[0], 'gs')
    pts = [[p] for p in pl.linspace(min(X), max(X), 100)]
    Yp_reg_fit = pl.dot(regression_fit_w, designMatrix(pts, order).T)
    Yp_finite = pl.dot(finite_diff_w, designMatrix(pts, order).T)
    Yp_opt = pl.dot(optimized_function_w, designMatrix(pts, order).T)
    Yp_analytical_reg = pl.dot(analytical_regularized_w, designMatrix(pts, order).T)
    Yp_numerical_reg = pl.dot(numerical_regularized_w, designMatrix(pts, order).T)
    pl.plot(pts, Yp_reg_fit.tolist()[0], 'b-',
            pts, Yp_finite.tolist(), 'g--',
            pts, Yp_opt.tolist(), 'r-.',
            pts, Yp_analytical_reg.tolist()[0], 'y:',
            pts, Yp_numerical_reg.tolist(), 'k.')
    pl.show()

    return np.asarray([regression_fit_w.flatten(),
                       finite_diff_w.flatten(),
                       optimized_function_w.flatten(),
                       analytical_regularized_w.flatten(),
                       numerical_regularized_w.flatten()])


def validate_models():
    X, Y = regressAData()
    wA = model_selection(X, Y, 1)
    X, Y = regressBData()
    wB = model_selection(X, Y, 1, wA)

    X, Y = validateData()
    phi = designMatrix(X, 1)
    args = (Y, phi)
    error = np.zeros(5)
    models = ["Gradient Descent", "Gradient Descent using Finite Differences Theorem", "Using scipy.fmin_bfgs",
              "With Ridge Regression", "with Ridge Regression using Numerical Method"]
    print("\nComparing all the models: Model Selection based on errors\n")
    for i in range(5):
        error[i] = sum_of_squared_errors(wB[i], *args)
        print(models[i], " : ", error[i])


if __name__ == "__main__":
    validate_models()
