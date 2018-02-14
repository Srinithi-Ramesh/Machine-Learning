from plotBoundary import *
import numpy as np
from cvxopt import *
from scipy import optimize


###################################################################################
# 1.1 Implement logistic regression with quadratic(L2) regularization
# using finite difference minimizer. Prediction function for linearly separable
# data is predictLR
# count_error is used to report the number of mistakes on the training data
###################################################################################
w_train = np.zeros((3,1))

def yt(w, x, y):
    return (dot(w[1:len(w)].T, x) + w[0]) * y


def elr(w, x, y):
    return math.log(1 + exp(-yt(w, x, y)[0]))


def logistic_regression_nll(w, *args):
    x, y, l = args
    NLL = 0
    for i in range(len(y)):
        NLL = NLL + elr(w, x[i], y[i])
    return NLL + l * dot(w[1:len(w)].T, w[1:len(w)])


def sigm(w, x, y):
    return 1 / (1 + exp(dot(y.T, (dot(x, w[1:len(w)]) + w[0]))))


def gradient_descent_finite_theorem(f, g, *args, initial_guess, step_size, convergence_criteria):
    error = math.inf
    curr = initial_guess
    count = 0
    while abs(error) > convergence_criteria:
        grad = g(f, curr, *args)
        updated_values = np.subtract(curr, tuple([step_size * x for x in grad]))
        curr_f = f(curr, *args)
        updated_f = f(updated_values, *args)
        error = curr_f - updated_f
        curr = updated_values
        count = count + 1
        if count % 1000 == 0:
            print(error)
    print("Total iterations: %s" % count)
    return curr


h = 0.01


def finite_differences_gradient(f, z, *args):
    try:
        length = len(z)
        z = np.asarray(z)
        grad = np.zeros((length, 1))
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


def predictLR(x):
    return 1 / (1 + exp(-(dot(x, w_train[1:3]) + w_train[0])[0]))


def count_error(X, Y):
    error_count = 0
    for i in range(len(Y)):
        pred_y = 1 if predictLR(X[i]) >= 0.5 else -1
        if pred_y != Y[i]:
            error_count += 1
    return error_count


#########################################################################################
# The regularization term lambda can be set here
#########################################################################################
l = 10

# The initial guess can be modified here
guess = np.empty((3, 1))
guess.fill(1)

data = 'ls'
print('======Training======')
train = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_train.csv')
X_train = train[:, 0:2]
Y_train = train[:, 2:3]
args = (X_train, Y_train, l)


# w_train = gradient_descent_finite_theorem(logistic_regression_nll,
#                                           finite_differences_gradient,
#                                           *args, initial_guess=guess, step_size=0.01,
#                                           convergence_criteria=0.0001)

# print("Training error in linearly separable data:", count_error(X_train, Y_train))

###########################################################################################
# 1.2 Effect of regularization term on logistic regression
###########################################################################################


def plotting_logistic_regression():
    data = 'ls'
    print('======Training======')
    train = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_train.csv')
    X_train = train[:, 0:2]
    Y_train = train[:, 2:3]
    args = (X_train, Y_train, l)

    w_train = gradient_descent_finite_theorem(logistic_regression_nll,
                                              finite_differences_gradient,
                                              *args, initial_guess=guess, step_size=0.01,
                                              convergence_criteria=0.0001)

    print("Weights on ls data:", w_train)

    print("Training error in linearly separable data:", count_error(X_train, Y_train))
    plotDecisionBoundary(X_train, Y_train, predictLR, [0.5], title='LR Train')

    print('======Validation======')
    validate = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_validate.csv')
    X_validate = validate[:, 0:2]
    Y_validate = validate[:, 2:3]
    print("Testing error in non-linearly separable data:", count_error(X_validate, Y_validate))
    plotDecisionBoundary(X_validate, Y_validate, predictLR, [0.5], title='LR Validate')

    data = 'nls'
    print('======Training======')
    train = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_train.csv')
    X_train = train[:, 0:2]
    Y_train = train[:, 2:3]
    args = (X_train, Y_train, l)

    w_train = gradient_descent_finite_theorem(logistic_regression_nll,
                                              finite_differences_gradient,
                                              *args, initial_guess=guess, step_size=0.01,
                                              convergence_criteria=0.0001)
    print("Weights on nls data:", w_train)
    print("Training error in non-linearly separable data:", count_error(X_train, Y_train))
    plotDecisionBoundary(X_train, Y_train, predictLR, [0.5], title='NLS Train')

    print('======Validation======')
    validate = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_validate.csv')
    X_validate = validate[:, 0:2]
    Y_validate = validate[:, 2:3]
    print("Testing error in non-linearly separable data:", count_error(X_validate, Y_validate))
    plotDecisionBoundary(X_validate, Y_validate, predictLR, [0.5], title='NLS Validate')

    data = 'nonlin'
    print('======Training======')
    train = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_train.csv')
    X_train = train[:, 0:2]
    Y_train = train[:, 2:3]
    args = (X_train, Y_train, l)

    w_train = gradient_descent_finite_theorem(logistic_regression_nll,
                                              finite_differences_gradient,
                                              *args, initial_guess=guess, step_size=0.01,
                                              convergence_criteria=0.0001)
    print("Weights on non-lin data:", w_train)
    print("Training error in non-linear data:", count_error(X_train, Y_train))
    plotDecisionBoundary(X_train, Y_train, predictLR, [0.5], title='Non Linear Train')

    print('======Validation======')
    validate = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_validate.csv')
    X_validate = validate[:, 0:2]
    Y_validate = validate[:, 2:3]
    print("Testing error in non-linear data:", count_error(X_validate, Y_validate))
    plotDecisionBoundary(X_validate, Y_validate, predictLR, [0.5], title='Non Linear Validate')


# print("Errors when there is no regularization:")
# l = 0
# plotting_logistic_regression()
# print("Errors when there is lambda = 10:")
# l = 10
# plotting_logistic_regression()
# print("Errors when there is lambda = 100:")
# l = 100
# plotting_logistic_regression()


###########################################################################################
# 1.3 Including Second order functions
###########################################################################################
l = 10


def second_order_kernel_fn(x):
    x0 = np.asarray(x[:, 0])
    x1 = np.asarray(x[:, 1])
    x2 = np.multiply(x0, x1)
    x3 = np.multiply(x0, x0)
    x4 = np.multiply(x1, x1)
    X = np.asarray((x0, x2, x2, x3, x4)).T
    return X


def second_order_kernel_for_prediction(x):
    x0 = np.asarray(x[0])
    x1 = np.asarray(x[1])
    x2 = np.multiply(x0, x1)
    x3 = np.multiply(x0, x0)
    x4 = np.multiply(x1, x1)
    X = np.asarray((x0, x2, x2, x3, x4)).T
    return X


w_train_with_kernel = np.zeros((6, 1))


def predict_with_kernel(x):
    return 1 / (1 + exp(-(dot(second_order_kernel_for_prediction(x).T,
                              w_train_with_kernel[1:len(w_train_with_kernel)]) +
                          w_train_with_kernel[0])[0]))


def count_error_kernel(X, Y):
    error_count = 0
    for i in range(len(Y)):
        pred_y = 1 if predict_with_kernel(X[i]) > 0.5 else -1
        if pred_y != Y[i]:
            error_count += 1
    return error_count


def plotting_logistic_regression_kernel():
    data = 'ls'
    print('======Training======')
    train = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_train.csv')
    X_train = train[:, 0:2]
    Y_train = train[:, 2:3]
    KX_train = second_order_kernel_fn(X_train)
    guess = np.zeros((6, 1))
    guess.fill(3)
    args = (KX_train, Y_train, l)

    w_train_with_kernel = gradient_descent_finite_theorem(logistic_regression_nll,
                                                          finite_differences_gradient,
                                                          *args, initial_guess=guess,
                                                          step_size=0.01,
                                                          convergence_criteria=0.0001)

    print("Weights on ls data with second order kernel:", w_train_with_kernel)

    print("Training error in linearly separable data with second order kernel:",
          count_error_kernel(X_train, Y_train))
    plotDecisionBoundary(X_train, Y_train, predict_with_kernel, [0.5], title='LR Train')

    print('======Validation======')
    validate = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_validate.csv')
    X_validate = validate[:, 0:2]
    Y_validate = validate[:, 2:3]
    print("Testing error in linearly separable data:", count_error_kernel(X_validate, Y_validate))
    plotDecisionBoundary(X_validate, Y_validate, predict_with_kernel, [0.5], title='LR Validate')

    data = 'nls'
    print('======Training======')
    train = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_train.csv')
    X_train = train[:, 0:2]
    Y_train = train[:, 2:3]
    KX_train = second_order_kernel_fn(X_train)
    args = (KX_train, Y_train, l)
    w_train_with_kernel = gradient_descent_finite_theorem(logistic_regression_nll,
                                                          finite_differences_gradient,
                                                          *args, initial_guess=guess, step_size=0.01,
                                                          convergence_criteria=0.0001)
    print("Weights on nls data:", w_train_with_kernel)
    print("Training error in non-linearly separable data with second order kernel:",
          count_error_kernel(X_train, Y_train))
    plotDecisionBoundary(X_train, Y_train, predict_with_kernel, [0.5], title='NLS Train')

    print('======Validation======')
    validate = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_validate.csv')
    X_validate = validate[:, 0:2]
    Y_validate = validate[:, 2:3]
    print("Testing error in non-linearly separable data with second order kernel:",
          count_error_kernel(X_validate, Y_validate))
    plotDecisionBoundary(X_validate, Y_validate, predict_with_kernel, [0.5], title='NLS Validate')

    data = 'nonlin'
    print('======Training======')
    train = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_train.csv')
    X_train = train[:, 0:2]
    Y_train = train[:, 2:3]
    KX_train = second_order_kernel_fn(X_train)
    args = (KX_train, Y_train, l)

    w_train_with_kernel = gradient_descent_finite_theorem(logistic_regression_nll,
                                                          finite_differences_gradient,
                                                          *args, initial_guess=guess,
                                                          step_size=0.01,
                                                          convergence_criteria=0.0001)
    print("Weights on non-lin data with second order kernel:", w_train_with_kernel)
    print("Training error in non-linear data with second order kernel:", count_error_kernel(X_train, Y_train))
    plotDecisionBoundary(X_train, Y_train, predict_with_kernel, [0.5], title='Non Linear Train')

    print('======Validation======')
    validate = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_validate.csv')
    X_validate = validate[:, 0:2]
    Y_validate = validate[:, 2:3]
    print("Testing error in non-linear data with second order kernel:", count_error_kernel(X_validate, Y_validate))
    plotDecisionBoundary(X_validate, Y_validate, predict_with_kernel, [0.5], title='Non Linear Validate')


# print("Results with kernel")
# plotting_logistic_regression_kernel()
###############################################################################################
# 2.1 Support Vector Machine Implementation
###############################################################################################
w_train_SVM = 0
b1 = 0
c = 10


def dual_svm(x, y):
    p1 = np.outer(y, y) * dot(x, x.T)
    p = matrix(p1)
    q = np.empty((x.shape[0], 1))
    q.fill(-1)
    q = matrix(q)
    g = matrix(-1 * np.eye(x.shape[0]))
    h = matrix(np.zeros(x.shape[0]))
    a = matrix(y.T)
    b = matrix(np.zeros(1))
    sol = solvers.qp(p, q, g, h, a, b)
    return np.asarray(sol['x'])


def dual_svm_nls(x, y, c):
    p1 = np.outer(y, y) * dot(x, x.T)
    p = matrix(p1)
    q = np.empty((x.shape[0], 1))
    q.fill(-1)
    q = matrix(q)
    g1 = matrix(-1 * np.eye(x.shape[0]))
    g2 = matrix(np.eye(x.shape[0]))
    g = matrix([g1, g2])
    h1 = matrix(np.zeros(x.shape[0]))
    h2 = np.empty((x.shape[0], 1))
    h2.fill(c)
    h2 = matrix(h2)
    h = matrix([h1, h2])
    a = matrix(y.T)
    b = matrix(np.zeros(1))
    sol = solvers.qp(p, q, g, h, a, b)
    return np.asarray(sol['x'])


def predictSVM(x):
    return dot(x, w_train_SVM.T) + b1[0]


def count_errorSVM(X, Y):
    error_count = 0
    for i in range(len(Y)):
        pred_y = -1 if predictSVM(X[i]) < 0 else 1
        if pred_y != Y[i]:
            error_count += 1
    return error_count


# data = 'ls'
# print('======Training======')
# train = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_train.csv')
# X_train = train[:, 0:2]
# Y_train = train[:, 2:3]
# alpha = dual_svm(X_train, Y_train)
# w_train_SVM = np.sum(alpha * Y_train * X_train, axis=0)
#
# for i in range(len(alpha)):
#     if alpha[i] > 1e-4:
#         b1 = Y_train[i] - np.dot(X_train[i], w_train_SVM)
#         break
# print("bias:", b1)
#
# print("Weights on ls data:", b1, w_train_SVM)
#
# print("Training error in linearly separable data:", count_errorSVM(X_train, Y_train))
# plotDecisionBoundary(X_train, Y_train, predictSVM, [0.5], title='LR Train')
#
# print('======Validation======')
# validate = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_validate.csv')
# X_validate = validate[:, 0:2]
# Y_validate = validate[:, 2:3]
# print("Testing error in non-linearly separable data:", count_errorSVM(X_validate, Y_validate))
# plotDecisionBoundary(X_validate, Y_validate, predictSVM, [0.5], title='LR Validate')
#
# data = 'nls'
# print('======Training======')
# train = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_train.csv')
# X_train = train[:, 0:2]
# Y_train = train[:, 2:3]
# alpha = dual_svm_nls(X_train, Y_train, c)
# w_train_SVM = np.sum(alpha * Y_train * X_train, axis=0)
#
# for i in range(len(alpha)):
#     if alpha[i] > 1e-4:
#         b1 = Y_train[i] - np.dot(X_train[i], w_train_SVM)
#         break
# print("bias:", b1)
# print("Weights on nls data:", w_train_SVM)
# print("Training error in non-linearly separable data:", count_errorSVM(X_train, Y_train))
# plotDecisionBoundary(X_train, Y_train, predictSVM, [0.5], title='NLS Train')
#
# print('======Validation======')
# validate = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_validate.csv')
# X_validate = validate[:, 0:2]
# Y_validate = validate[:, 2:3]
# print("Testing error in non-linearly separable data:", count_errorSVM(X_validate, Y_validate))
# plotDecisionBoundary(X_validate, Y_validate, predictSVM, [0.5], title='NLS Validate')
#
# data = 'nonlin'
# print('======Training======')
# train = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_train.csv')
# X_train = train[:, 0:2]
# Y_train = train[:, 2:3]
# args = (X_train, Y_train, l)
# alpha = dual_svm_nls(X_train, Y_train, c)
# w_train_SVM = np.sum(alpha * Y_train * X_train, axis=0)
#
# for i in range(len(alpha)):
#     if alpha[i] > 1e-4:
#         b1 = Y_train[i] - np.dot(X_train[i], w_train_SVM)
#         break
# print("bias:", b1)
#
# print("Weights on non-lin data:", w_train_SVM)
# print("Training error in non-linear data:", count_errorSVM(X_train, Y_train))
# plotDecisionBoundary(X_train, Y_train, predictSVM, [0.5], title='Non Linear Train')
#
# print('======Validation======')
# validate = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_validate.csv')
# X_validate = validate[:, 0:2]
# Y_validate = validate[:, 2:3]
# print("Testing error in non-linear data:", count_errorSVM(X_validate, Y_validate))
# plotDecisionBoundary(X_validate, Y_validate, predictSVM, [0.5], title='Non Linear Validate')


###############################################################################################
# 3.1 Kernel SVM
###############################################################################################
def calculate_slack(weight, *args):
    w = weight[1:len(weight)]
    b = weight[0]
    c, x, y = args
    slack = np.zeros(len(y))
    for i in range(len(y)):
        new_slack = 1 - (y[i] * (dot(w, x[i]) + b))[0]
        slack[i] = 0 if new_slack < 0 else new_slack
    return slack


def opt_func(weight, *args):
    c, x, y = args
    return 0.5 * dot(weight[1: len(weight)], weight[1: len(weight)].T) + \
           c * np.sum(calculate_slack(weight, *args))


def svm_primal(x, y, c):
    def constaint_1(weight, *args_l):
        w_l = weight[1:len(weight)]
        b_l = weight[0]
        c, x_l, y_l = args_l
        val_l = np.zeros(len(y_l))
        slack_l = calculate_slack(weight, *args_l)
        for i in range(len(val_l)):
            val_l[i] = ((np.sum(x_l[0] * w_l) + b_l) * y_l[0] - 1 - slack_l[0])[0]
        return val_l

    w = np.ones((3, 1))
    args = c, x, y
    cons = ({'type': 'ineq', 'fun': constaint_1, 'args': args},
            {'type': 'ineq', 'fun': calculate_slack, 'args': args})
    res = optimize.minimize(opt_func, w, args=args, constraints=cons)
    return res


def polynomial_kernel(x, y, p=2):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=1.0):
    return np.exp((-(np.linalg.norm(x-y, axis=None)**2))/(2*sigma**2))


def dual_svm_kernel(x, y, c, kernel):
    K = np.zeros((len(y), len(y)))
    for i in range(len(y)):
        for j in range(len(y)):
            K[i, j] = kernel(x[i], x[j])
    p1 = np.outer(y, y) * K
    p = matrix(p1)
    q = np.empty((x.shape[0], 1))
    q.fill(-1)
    q = matrix(q)
    g1 = matrix(-1 * np.eye(x.shape[0]))
    g2 = matrix(np.eye(x.shape[0]))
    g = matrix([g1, g2])
    h1 = matrix(np.zeros(x.shape[0]))
    h2 = np.empty((x.shape[0], 1))
    h2.fill(c)
    h2 = matrix(h2)
    h = matrix([h1, h2])
    a = matrix(y.T)
    b = matrix(np.zeros(1))
    sol = solvers.qp(p, q, g, h, a, b)
    return np.asarray(sol['x'])


def predictSVM_polynomial_kernel(x):
    val = 0
    for i in range(len(alpha)):
        val += alpha[i] * Y_train[i] * polynomial_kernel(X_train[i], x)
    val = np.sign(val + b1[0])
    return val


def predictSVM_gaussian_kernel(x):
    val = 0
    for i in range(len(alpha)):
        val += alpha[i] * Y_train[i] * gaussian_kernel(X_train[i], x)
    val = np.sign(val + b1[0])
    return val


def count_error_SVM_polynomial_kernel(X, Y):
    error_count = 0
    for i in range(len(Y)):
        pred_y = -1 if predictSVM_polynomial_kernel(X[i]) < 0 else 1
        if pred_y != Y[i]:
            error_count += 1
    return error_count


def count_error_SVM_gaussian_kernel(X, Y):
    error_count = 0
    for i in range(len(Y)):
        pred_y = -1 if predictSVM_gaussian_kernel(X[i]) < 0 else 1
        if pred_y != Y[i]:
            error_count += 1
    return error_count


c = 50

data = 'ls'
print('======Training======')
train = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_train.csv')
X_train = train[:, 0:2]
Y_train = train[:, 2:3]
alpha = dual_svm_kernel(X_train, Y_train, c, polynomial_kernel)
w_train_SVM = np.sum(alpha * Y_train * X_train, axis=0)

for i in range(len(alpha)):
    if alpha[i] > 1e-4:
        b1 = Y_train[i] - np.dot(X_train[i], w_train_SVM)
        break
print("bias:", b1)

print("Weights on ls data:", b1, w_train_SVM)

print("Training error in linearly separable data:", count_error_SVM_polynomial_kernel(X_train, Y_train))
plotDecisionBoundary(X_train, Y_train, predictSVM_polynomial_kernel, [0.5], title='LR Train Polynomial Kernel')

print('======Validation======')
validate = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_validate.csv')
X_validate = validate[:, 0:2]
Y_validate = validate[:, 2:3]
print("Testing error in linearly separable data:", count_error_SVM_polynomial_kernel(X_validate, Y_validate))
plotDecisionBoundary(X_validate, Y_validate, predictSVM_polynomial_kernel, [0.5], title='LR Validate Polynomial Kernel')

# print('======Training======')
# train = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_train.csv')
# X_train = train[:, 0:2]
# Y_train = train[:, 2:3]
# alpha = dual_svm_kernel(X_train, Y_train, c, gaussian_kernel)
# w_train_SVM = np.sum(alpha * Y_train * X_train, axis=0)
#
# for i in range(len(alpha)):
#     if alpha[i] > 1e-4:
#         b1 = Y_train[i] - np.dot(X_train[i], w_train_SVM)
#         break
# print("bias:", b1)
#
# print("Weights on ls data:", b1, w_train_SVM)
#
# print("Training error in linearly separable data:", count_error_SVM_gaussian_kernel(X_train, Y_train))
# plotDecisionBoundary(X_train, Y_train, predictSVM_gaussian_kernel, [0.5], title='LR Train Gaussian Kernel')
#
# print('======Validation======')
# validate = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_validate.csv')
# X_validate = validate[:, 0:2]
# Y_validate = validate[:, 2:3]
# print("Testing error in non-linearly separable data:", count_error_SVM_gaussian_kernel(X_validate, Y_validate))
# plotDecisionBoundary(X_validate, Y_validate, predictSVM_gaussian_kernel, [0.5], title='LR Validate Gaussian Kernel')
# #
# #######################################################################################################################

data = 'nls'
print('======Training======')
train = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_train.csv')
X_train = train[:, 0:2]
Y_train = train[:, 2:3]
alpha = dual_svm_kernel(X_train, Y_train, c, polynomial_kernel)
w_train_SVM = np.sum(alpha * Y_train * X_train, axis=0)

for i in range(len(alpha)):
    if alpha[i] > 1e-4:
        b1 = Y_train[i] - np.dot(X_train[i], w_train_SVM)
        break
print("bias:", b1)
print("Weights on nls data:", w_train_SVM)
print("Training error in non-linearly separable data:", count_error_SVM_polynomial_kernel(X_train, Y_train))
plotDecisionBoundary(X_train, Y_train, predictSVM_polynomial_kernel, [0.5], title='NLS Train Polynomial kernel')

print('======Validation======')
validate = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_validate.csv')
X_validate = validate[:, 0:2]
Y_validate = validate[:, 2:3]
print("Testing error in non-linearly separable data:", count_error_SVM_polynomial_kernel(X_validate, Y_validate))
plotDecisionBoundary(X_validate, Y_validate, predictSVM_polynomial_kernel, [0.5],
                     title='NLS Validate polynomial kernel')
#
# print('======Training======')
# train = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_train.csv')
# X_train = train[:, 0:2]
# Y_train = train[:, 2:3]
# alpha = dual_svm_kernel(X_train, Y_train, c, gaussian_kernel)
# w_train_SVM = np.sum(alpha * Y_train * X_train, axis=0)
#
# for i in range(len(alpha)):
#     if alpha[i] > 1e-4:
#         b1 = Y_train[i] - np.dot(X_train[i], w_train_SVM)
#         break
# print("bias:", b1)
# print("Weights on nls data:", w_train_SVM)
# print("Training error in non-linearly separable data:", count_error_SVM_gaussian_kernel(X_train, Y_train))
# plotDecisionBoundary(X_train, Y_train, predictSVM_gaussian_kernel, [0.5], title='NLS Train Gaussian kernel')
#
# print('======Validation======')
# validate = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_validate.csv')
# X_validate = validate[:, 0:2]
# Y_validate = validate[:, 2:3]
# print("Testing error in non-linearly separable data:", count_error_SVM_gaussian_kernel(X_validate, Y_validate))
# plotDecisionBoundary(X_validate, Y_validate, predictSVM_gaussian_kernel, [0.5], title='NLS Validate Gaussian kernel')
# #
# ########################################################################################################################
#
#
data = 'nonlin'
print('======Training======')
train = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_train.csv')
X_train = train[:, 0:2]
Y_train = train[:, 2:3]
args = (X_train, Y_train, l)
alpha = dual_svm_kernel(X_train, Y_train, c, polynomial_kernel)
w_train_SVM = np.sum(alpha * Y_train * X_train, axis=0)

for i in range(len(alpha)):
    if alpha[i] > 1e-4:
        b1 = Y_train[i] - np.dot(X_train[i], w_train_SVM)
        break
print("bias:", b1)

print("Weights on non-lin data:", w_train_SVM)
print("Training error in non-linear data:", count_error_SVM_polynomial_kernel(X_train, Y_train))
plotDecisionBoundary(X_train, Y_train, predictSVM_polynomial_kernel, [0.5], title='Non Linear Train Polynomial Kernel')

print('======Validation======')
validate = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_validate.csv')
X_validate = validate[:, 0:2]
Y_validate = validate[:, 2:3]
print("Testing error in non-linear data:", count_error_SVM_polynomial_kernel(X_validate, Y_validate))
plotDecisionBoundary(X_validate, Y_validate, predictSVM_polynomial_kernel, [0.5],
                     title='Non Linear Validate Polynomial kernel')

# print('======Training======')
# train = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_train.csv')
# X_train = train[:, 0:2]
# Y_train = train[:, 2:3]
# args = (X_train, Y_train, l)
# alpha = dual_svm_kernel(X_train, Y_train, c, gaussian_kernel)
# w_train_SVM = np.sum(alpha * Y_train * X_train, axis=0)
#
# for i in range(len(alpha)):
#     if alpha[i] > 1e-4:
#         b1 = Y_train[i] - np.dot(X_train[i], w_train_SVM)
#         break
# print("bias:", b1)
#
#
# print("Weights on non-lin data:", w_train_SVM)
# print("Training error in non-linear data:", count_error_SVM_gaussian_kernel(X_train, Y_train))
# plotDecisionBoundary(X_train, Y_train, predictSVM_gaussian_kernel, [0.5], title='Non Linear Train Gaussian Kernel')
#
# print('======Validation======')
# validate = loadtxt('/home/srinithi/NEU/HW2 code/data/data_' + data + '_validate.csv')
# X_validate = validate[:, 0:2]
# Y_validate = validate[:, 2:3]
# print("Testing error in non-linear data:", count_error_SVM_gaussian_kernel(X_validate, Y_validate))
# plotDecisionBoundary(X_validate, Y_validate, predictSVM_gaussian_kernel, [0.5], title='Non Linear Validate Gaussian kernel')
