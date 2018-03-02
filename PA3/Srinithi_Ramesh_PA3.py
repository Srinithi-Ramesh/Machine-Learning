import math
from numpy import *
from plotGauss2D import *
from em_test import *
from scipy.stats import multivariate_normal

xyz = 0

########################################################################################################################
# 1. EM Algorithm
########################################################################################################################

########################################################################################################################
# 1.1 Implement EM for Gaussian Mixtures as described in Murphy (section 11.4.21). Your program
# will need an input data set, initial mixture parameters and a convergence threshold (pick your
# favorite way of deciding convergence). The file em_test has a simple skeleton for reading
# data and plotting results. Note that the plots show mixture components as ellipses at two
# standard deviations.
########################################################################################################################


def initial_covariance_matrix(n):
    return array([[1, 1], [1, 1]])


def initial_mean():
    global xyz
    if xyz == 0:
        xyz += 1
        return asarray([1, -1])
    if xyz == 1:
        xyz += 1
        return asarray([-1, 3])
    if xyz == 2:
        xyz += 1
        return asarray([2, -2])
    if xyz == 3:
        xyz += 1
        return asarray([-2, 2])
    else:
        xyz += 1
        return asarray([-1, -1])


def initial_class_prior(k):
    prior = empty(k)
    prior.fill(1 / k)
    return prior


def multivar_normal_distr(x, mu, cov_mat):
    return multivariate_normal.pdf(x, mu, cov_mat, True)


def e_step(i, k, class_prob, x, mu, cov_mat):
    n = 0
    for j in range(len(class_prob)):
        n += class_prob[j] * multivar_normal_distr(x[i], mu[j], cov_mat[j])
    r_ik = (class_prob[k] * multivar_normal_distr(x[i], mu[k], cov_mat[k])) / n
    return r_ik


def m_step(r, k, x, cov_mat_type='diagonal'):
    r_sum = 0
    for j in range(len(x)):
        r_sum += r[j][k]
    new_class_prob = r_sum / len(x)
    new_mean_k = zeros((1, len(x[0])))
    for i in range(len(x)):
        new_mean_k += r[i][k] * x[i]
    new_mean_k = new_mean_k / r_sum
    new_cov_mat = zeros((len(x[0]), len(x[0])))
    for i in range(len(x)):
        new_cov_mat += r[i][k] * matmul(asarray([x[i]]).T, asarray([x[i]]))
    new_cov_mat = (new_cov_mat / r_sum) - matmul(new_mean_k.T, new_mean_k)
    if cov_mat_type == 'diagonal':
        new_cov_mat = diag(diag(new_cov_mat))
    return new_class_prob, new_mean_k[0], new_cov_mat


########################################################################################################################
# 1.2 To avoid numerical underflow problems, you will want to compute log likelihood. Note,
# however, that for a mixture distribution you will need to compute the log of a sum2
# . Look up
# the “logsumexp trick” to see how to deal with this. You can find logsumexp in Numpy. In
# any case, the idea is simple (once you see it).
########################################################################################################################


def lse(a):
    am = max(a)
    exp_sum = 0
    for i in range(len(a)):
        exp_sum += exp(a[i] - am)
    return am + log(exp_sum)


def is_invertible(mat):
    return mat.shape[0] == mat.shape[1] and linalg.matrix_rank(mat) == mat.shape[0]


def ensure_invertible(mat, multi_axis=0):
    if is_invertible(mat):
        return mat

    if multi_axis == 0:
        mat[0][0] += 1e-3
    else:
        mat[0][0] += 1e-4
        mat[0][1] += 1e-4
        mat[1][0] += 1e-4

    if not is_invertible(mat):
        return ensure_invertible(mat, 1)
    return mat


def log_gaussian(x, mu, cov_mat):
    cov_mat = ensure_invertible(cov_mat)

    return (-0.5 * matmul((subtract(x, mu)).T,
                          (matmul(linalg.inv(cov_mat),
                                  subtract(x, mu))))) - \
           (len(x) / 2) * log(2 * pi) -\
           0.5 * log(1e-15 if linalg.det(cov_mat) <= 0 else linalg.det(cov_mat))


def log_GMM(x, mu, cov_mat, class_prob):
    a = zeros((len(class_prob), 1))
    for j in range(len(class_prob)):
        a[j] = log_gaussian(x, mu[j], cov_mat[j]) + log(class_prob[j])
    return lse(a)


def e_step_log(i, k, class_prob, x, mu, cov_mat):
    r_ik = exp(log_gaussian(x[i], mu[k], cov_mat[k]) + log(class_prob[k]) - log_GMM(x[i], mu, cov_mat, class_prob))
    return r_ik


########################################################################################################################
# 1.3 Describe the behavior of your algorithm on all (non-mystery) training data sets provided as
# you vary (a) the number of components in the mixture, (b) the initial mixture parameters, (c)
# the convergence parameter and (d) the choice of small/large. Report the log-likelihoods and
# show selected plots of good and bad performance.
########################################################################################################################


def testing(x, K, class_prior, cov_mat, mean):
    r = zeros((len(x), K))
    for i in range(len(x)):
        for k in range(K):
            r[i][k] = e_step_log(i, k, class_prior, x, mean, cov_mat)
    sum_1 = 0
    for i in range(len(x)):
        for k in range(K):
            sum_1 += r[i][k] * log(class_prior[k])
    sum_2 = 0
    for i in range(len(x)):
        for k in range(K):
            pdf_val = multivar_normal_distr(x[i], mean[k], cov_mat[k])
            pdf_val = 1e-15 if pdf_val == 0 else pdf_val # To avoid log(0)
            sum_2 += r[i][k] * log(pdf_val)
    log_likelihood = sum_1 + sum_2
    return log_likelihood


########################################################################################################################
# 2. Variations:
########################################################################################################################

########################################################################################################################
# 2.1 . Modify your implementation of the EM algorithm so that it can build two types of models:
# ones with general covariance matrices and ones with diagonal matrices. Note that in each case
# different components have different means. Explain (in math, not code) the difference in the
# EM algorithm for these two variants
########################################################################################################################


def driver_em(x, K, max_itr, logsumtrick=False, cov_mat_type='diagonal'):
    itr = 0
    new_mean = zeros((K, len(x[0])))
    new_cov_mat = zeros((K, len(x[0]), len(x[0])))
    new_class_prior = zeros((K, 1))
    mean = k_means(x, K)
    cov_mat = zeros((K, len(x[0]), len(x[0])))
    cov_mat[0] = array([[1, 0], [0, 1]])
    cov_mat[1] = array([[1, 0], [0, 1]])
    for i in range(K):
        cov_mat[i] = initial_covariance_matrix(len(x[0]))
    class_prior = initial_class_prior(K)
    while itr < max_itr:
        r = zeros((len(x), K))
        for i in range(len(x)):
            for k in range(K):
                if(logsumtrick):
                    r[i][k] = e_step_log(i, k, class_prior, x, mean, cov_mat)
                else:
                    r[i][k] = e_step(i, k, class_prior, x, mean, cov_mat)
        for k in range(K):
            mstep = m_step(r, k, x, cov_mat_type)
            new_mean[k] = mstep[1]
            new_cov_mat[k] = mstep[2]
            new_class_prior[k] = mstep[0]
        mean = new_mean
        cov_mat = new_cov_mat
        class_prior = new_class_prior
        itr = itr + 1
    return mean, cov_mat, class_prior


########################################################################################################################
# 2.2 Implement the K-means algorithm as a way to get an initial estimate of the mixture components
########################################################################################################################

def euclidean_distance(x, y, ax=1):
    return linalg.norm(x - y, axis=ax)


def k_means(x, K):
    mean = zeros((K, len(x[0])))
    for i in range(K):
        mean[i, :] = initial_mean()
    clusters = zeros((K, len(x[0])))
    cluster_size = zeros(K)
    new_mean = zeros((K, len(x[0])))
    old_mean = new_mean
    while euclidean_distance(old_mean, mean, None) != 0:
        for i in range(len(x)):
            dist = euclidean_distance(x[i], mean)
            cluster = argmin(dist)
            cluster_size[cluster] += 1
            clusters[cluster] += x[i]
        for k in range(K):
            new_mean[k] = clusters[k] / cluster_size[k]
        old_mean = mean
        mean = new_mean
    return mean


def print_params(mean, cov, class_prior):
    for i in range(len(mean)):
        print("m%d=[%.5f,\n   %.5f]\n" % (i + 1, mean[i][0], mean[i][1]))
    print("\n")
    for i in range(len(cov)):
        print("cov%d=[%.5f\t%.5f\n      %.5f\t%.5f]\n" % (i + 1, cov[i][0][0], cov[i][0][1], cov[i][1][0], cov[i][1][1]))
    print("\n")
    for i in range(len(class_prior)):
        print("pi%d=%.5f\n\n" % (i + 1, class_prior[i][0]))


def read_data(filename):
    data = '/home/srinithi/PycharmProjects/Machine Learning/PA_3/data/%s' % filename
    return loadtxt(data + ".txt")


def evaluate_model(dataset_number=1, k=2, cov_mat_type='full', iter=15):
    X = read_data('data_%d_small' % dataset_number)
    mean, cov_mat, class_prior = driver_em(X, k, iter, cov_mat_type=cov_mat_type)

    print_params(mean, cov_mat, class_prior)

    plotMOG(X, [MOG(pi=class_prior[i],
                    mu=mean[i],
                    var=cov_mat[i])
                for i in range(k)], "Small_%d" % dataset_number)
    print("train_log likelihood=", testing(X, k, class_prior, cov_mat, mean) / len(X))

    test_X = read_data('data_%d_large' % dataset_number)
    plotMOG(test_X, [MOG(pi=class_prior[i],
                         mu=mean[i],
                         var=cov_mat[i])
                     for i in range(k)], "Large_%d" % dataset_number)
    print("test_log likelihood=", testing(test_X, k, class_prior, cov_mat, mean) / len(test_X))


########################################################################################################################
# 2.3 Explore the performance of these algorithm variants using similar tests to what you did on
# the original algorithm.
########################################################################################################################



########################################################################################################################
# 3. Model Selection:
########################################################################################################################

########################################################################################################################
# 3.1 Construct a candidate set of models for each of the data sets that differ on (a) the choices of
# covariance matrices (diagonal vs. general) and (b) the number of mixture components (1 – 5).
# Run the models for each small (non-mystery) training set based on average log likelihood
# from applying EM to the training set. You will need to decide exactly how to use EM (how to
# initialize, whether to run multiple times, etc); document your choices. Compare your results
# to the ranking of the models on the large “test” sets. Explain your findings.
########################################################################################################################

# For each possible choice of k
# – Randomly split the data into training and test set
# – Learn the GMM model using the training data and compute the log‐likelihood on test data
# – Repeat this multiple times to get a stable estimate of the test log‐likelihood
# – Select k that maximizes the test log‐likelihood
def get_data_split(n, k):
    split = int(floor(n / k))
    splits = []
    count = 0
    for i in range(k):
        splits.append([])
        for j in range(split):
            splits[i].append((j + count))
            if (j + count) == (n - 1):
                break
        count += split
    return splits


def cross_validate(x, folds, clusters, cov_mat_type, max_iter=15):
    splits = get_data_split(len(x), folds)
    log_likelihood = []
    mean = None
    cov_mat = None
    class_prior = None

    for i in range(folds):
        train_data = []
        for j in range(len(x)):
            if j not in splits[i]:
                train_data.append(x[j])
        test_data = [x[j] for j in splits[i]]
        mean, cov_mat, class_prior = driver_em(train_data, clusters, max_iter, cov_mat_type=cov_mat_type)
        log_likelihood.append(testing(test_data, clusters, class_prior, cov_mat, mean)[0] / len(test_data))
    return average(log_likelihood), mean, cov_mat, class_prior


def rank_models(data):
    best_mean = None
    best_cov_mat = None
    best_class_prior = None
    best_cluster = None
    best_cov_mat_type = None
    best_llh = -Infinity
    print("Log Likelihood:")
    print("leave-one-out Cross Validation")
    for i in range(2, 6):
        for type in ["diagonal", "full"]:
            llh, mean, cov_mat, class_prior = cross_validate(data, len(data) - 1, i, type)
            print("%d Clusters with %s covariance matrix : %f" %
                  (i, type, llh))
            if llh > best_llh:
                best_llh = llh
                best_cluster = i
                best_cov_mat_type = type
                best_mean = mean
                best_cov_mat = cov_mat
                best_class_prior = class_prior
    print("\n")
    print("6-Fold Cross Validation")
    for i in range(2, 6):
        for type in ["diagonal", "full"]:
            llh, mean, cov_mat, class_prior = cross_validate(data, 6, i, type)
            print("%d Clusters with %s covariance matrix : %f" %
                  (i, type, llh))
            if llh > best_llh:
                best_llh = llh
                best_cluster = i
                best_cov_mat_type = type
                best_mean = mean
                best_cov_mat = cov_mat
                best_class_prior = class_prior
    print("\n")
    return best_cluster, best_cov_mat_type, best_mean, best_cov_mat, best_class_prior


def run_for_all_datasets():
    print('##################data_1#################')
    data = read_data('data_1_small')
    test_data = read_data('data_1_large')
    cluster, type, mean, cov_mat, class_prior = rank_models(data)
    print('######Best Params######\n' +
          'Clusters: \t%d' % cluster +
          'Convergence Matrix Type: \t%s\n\n' % type
          )
    print('Log Likelihood for large dataset: %f' %
          (testing(test_data, cluster, class_prior, cov_mat, mean) / len(test_data)))

    print('##################data_2_small#################')
    data = read_data('data_2_small')
    test_data = read_data('data_2_large')
    cluster, type, mean, cov_mat, class_prior = rank_models(data)
    print('######Best Params######\n' +
          'Clusters: \t%d' % cluster +
          'Convergence Matrix Type: \t%s\n\n' % type
          )
    print('Log Likelihood for large dataset: %f' %
          (testing(test_data, cluster, class_prior, cov_mat, mean) / len(test_data)))

    print('##################data_3_small#################')
    data = read_data('data_3_small')
    test_data = read_data('data_3_large')
    cluster, type, mean, cov_mat, class_prior = rank_models(data)
    print('######Best Params######\n' +
          'Clusters: \t%d' % cluster +
          'Convergence Matrix Type: \t%s\n\n' % type
          )
    print('Log Likelihood for large dataset: %f' %
          (testing(test_data, cluster, class_prior, cov_mat, mean) / len(test_data)))


def evaluate_final_model(clusters, cov_mat_type, train_file, test_file):
    train_data = read_data(train_file)
    test_data = read_data(test_file)

    mean, cov_mat, class_prior = driver_em(train_data, clusters, cov_mat_type=cov_mat_type, max_itr=15)

    log_likelihood = testing(test_data, clusters, class_prior, cov_mat, mean)
    print_params(mean, cov_mat, class_prior)
    print("Log Likelihood for Mystery Test : %f" % (log_likelihood / len(test_data)))


if __name__ == "__main__":
    # Running the model for all the datasets with all possible configuration
    for n in [1, 2, 3]:
        print("#######Dataset %d#######" % n)
        for k in range(2, 6):
            for type in ["diagonal", "full"]:
                print("%d Clusters with %s covariance matrix" % (k, type))
                evaluate_model(n, k, type)

    # Cross validation for all datasets
    run_for_all_datasets()

    # Running the model for the Mystery dataset
    for k in range(2, 6):
        for type in ["diagonal", "full"]:
                print("%d Clusters with %s covariance matrix" % (k, type))
                evaluate_final_model(k, type, 'mystery_1', 'mystery_2')
