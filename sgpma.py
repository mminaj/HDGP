import numpy as np
import random
import time

from Codes_2020.Statics import *
from kriging_gpr.src.kriging_model.interface.OK_Rmodel_kd_nugget import *
from kriging_gpr.src.kriging_model.interface.OK_Rpredict import *
from kriging_gpr.src.kriging_model.utils.normalize_data import *
from kriging_gpr.example_files.sampling import *
from kriging_gpr.example_files.calculate_robustness import *
from kriging_gpr.example_files.testFunction import *
from test_functions import *
from gpmcmc import *


# def stochastic_gen(y_det, nb_rep):
#     k = y_det.shape[0]
#     y_rep = np.zeros((nb_rep, k))
#     for i in range(nb_rep):
#         noise = np.random.rand(y_det.shape[0], y_det.shape[1]) * np.sqrt(0.8) # (k*1)
#         y_rep[i, :] = y_det.flatten() + noise.flatten() # y ~ N(y_det, 0.8)
#     y = np.mean(y_rep, axis=0).reshape((k, 1))
#     sigma_e = np.ones((k, 1)) * 0.8
#     return {'y': y, 'sigma_e': sigma_e}


# def stochastic_gen(y_det, nb_rep):
#     k = y_det.shape[0]
#     # noise = noise_conf_2019(y_det, noise_nb=1)
#     noise_std = np.zeros((k, 1)) + np.sqrt(0.8)
#     replications = np.zeros((k, nb_rep))  # (100, 15)
#     for i in range(nb_rep):
#         replications[:, i] = np.random.normal(y_det[:, 0], noise_std[:, 0])
#     # print(replications.shape) # (100, 15)
#     y = np.mean(replications, axis=1)
#     y = y.reshape((y.shape[0], 1))
#     sigma_e = np.var(replications, axis=1) / nb_rep
#     sigma_e = sigma_e.reshape((sigma_e.shape[0], 1))
#     # print(sigma_e.shape)
#     return {'y': y, 'sigma_e': sigma_e}


# def mnek_prediction(X, y, sigma_e, regr_model, corr_model, X_test):
#     """
#
#     :param X: original data of k*d
#     :param y: original data k*1
#     :param sigma_e: simulation output variances of k*1
#     :param regr_model:
#     :param corr_model:
#     :param X_test: k_test*d
#     :return:
#     """
#     model_dict = {'x': X, 'y': y, 'sigma_e': sigma_e,
#                   'regr_model': regr_model, 'corr_model': corr_model,
#                   'optimization_function': optimization_scipy}
#     model = MNEK(model_dict)
#     # start = time.time()
#     model.generate_model()
#     # runtime = time.time() - start
#     pred = model.predict(X_test)
#     return pred


# def ki_gen(k, d, n):
#     """
#     generate a set of ki's, sum of ki's = k, each ki at least 10*d
#     :param k: original number of data
#     :param d: dimension of original data
#     :param n: number of subsamples
#     :return: nparray n*1
#     """
#     ki_list = np.zeros([n, 1])
#     rmd = k - 10 * k
#     for i in range(n):
#         ki_tmp = random.choice(rmd)
#         rmd -= ki_tmp
#
#     I_S = np.random.permutation(k)
#
#     return


def low_size_low_dim_subset_gen(X, Y, n, ki, di, emb='pca'):
    """

    :param type_of_reduction:
    :param X: original data of k*d
    :param Y: original data k*1
    :param n: number of low embeddings to generate
    :param ki: nparray of sizes of subsets, ki<d, i=1,...,n
    :param di: nparray of dimension of low embedding, di<d, i=1,...,n
    :return: a list of all the n generated subsets
    """
    np.random.seed(520)
    k = X.shape[0]  # the complete index set list {0,..,k-1} of length k
    I_S = np.random.permutation(k)  # a list of disordered indices of length k
    X_gen = list()
    y_gen = list()
    for i in range(n):
        k_current = ki[i]
        I_Si = I_S[:k_current]
        Xi_tilt, Yi = subsamples(X, Y, I_Si)
        if emb == 'random':
            Xi = random_embed(Xi_tilt, di[i])
        else:
            Xi = PCA_reduction(Xi_tilt, di[i])
        X_gen.append(Xi)
        y_gen.append(Yi)
        I_S = np.delete(I_S, np.s_[:k_current], axis=0)  # remove the ki[i] indices from the total index set
        # which has already been used to generate subset in the current loop
    return X_gen, y_gen


def subsamples(X, Y, I_Si):
    """

    :param X: original data k*d
    :param Y: original data k*1
    :param I_Si: reduced index set
    :return: a tuple of subset X and y with corresponding index set
    """
    ki = len(I_Si)
    k = X.shape[0]
    M_Si = np.zeros((ki, k))
    for l in range(ki):
        for j in range(k):
            if I_Si[l] == j:
                M_Si[l, j] = 1
            else:
                M_Si[l, j] = 0
    # print("???????", M_S_i.shape, X.shape)
    Xi_tilt = np.matmul(M_Si, X)
    Yi = np.matmul(M_Si, Y)
    return Xi_tilt, Yi


def random_embed(Xi_tilt, di):
    """

    :param Xi_tilt:
    :param di:
    :return:
    """
    np.random.seed(520)
    d = Xi_tilt.shape[1]
    I_Di = np.random.choice(np.arange(d), di, replace=False)
    M_Di = np.zeros((di, d))
    for l in range(di):
        for j in range(d):
            if I_Di[l] == j:
                M_Di[l, j] = 1
            else:
                M_Di[l, j] = 0
    X_i = np.matmul(Xi_tilt, np.matrix.transpose(M_Di))  # k_i*d_i
    return X_i


# def pls_reduction(X, y, p):
#     if type(X) is not list:
#         return PLSRegression(n_components=p).fit_transform(X, y)
#     else:
#         return [PLSRegression(n_components=p).fit_transform(i, y[i,:]) for i in X]


def weights(k, d, ki, di, eta):
    pi = np.multiply((ki / k) ** 2, (di / d) ** eta)
    return pi * 1 / np.sum(pi, axis=0)


# same size of subsets but allocate the points randomly - sampling without replacement until filling up the urn -
# guarantee all models/subsets have same number  of points
# def low_size_low_dim_subset_gen(X, Y, n, di, using_pca=True):
#     """
#
#     :param X: original data of k*d
#     :param Y: original data k*1
#     :param n: number of low embeddings to generate
#     :param ki: number of points in each embedding, ki<k, i=1, ..., n
#     :param di: dimension of low embedding, di<d, i=1,...,n
#     :param using_pca: boolean, if False, using random embedding
#     :return: a list of all the n generated subsets
#     """
#     k = X.shape[0]
#     d = X.shape[1]
#     I_S = range(k) # the complete index set list {0,..,k-1} of length k
#     ki = 10 * d  # fix each subset size to have 10*d sample points
#     X_gen = list()
#     for i in range(n):
#         I_Si = np.random.choice(I_S, ki, replace=False)  # choose k_i samples from {1, ..., k}
#         X_i_tilt, Y_i = subsamples(X, Y, I_Si)
#         if using_pca:
#             X_i = Statics.PCA_reduction(X_i_tilt, di)
#         else:
#             X_i = random_embed(X_i_tilt, di)
#         X_gen.append(X_i)
#         I_S = np.delete(I_S, I_Si)  # remove the k_i indices from the total index set
#     return X_gen


# choose minimal number of points in each sample 10*dim, the remaining points put in the algorithm
# def low_dim_subset_gen(X, Y, n, d_i, using_PCA=True):
#     k = X.shape[0]
#     d = X.shape[1]
#     for i in range(1, n + 1):  # given n subsets, loop over each of the n subsets and assign k_i number of samples for
#         # each subsets
#         k_i = np.random.randint(1, k + 1)
#         I_S_i = np.random.choice(np.arange(1, k + 1), k_i)  # choose k_i samples from [1, 2, 3, ..., k]
#         X_i_tilt, Y_i = subsamples(X, Y, I_S_i)
#         if using_PCA:
#             X_i = pca_embed(X_i_tilt, d_i)
#         else:
#             X_i = random_embed(X_i_tilt, d_i)
#         k -= k_i
#     return X_i


# def pca_embed(X_i_tilt, d_i):
#     d = X.shape[1]
#     C_i = np.cov(X_i_tilt)
#     eig_vec, eig_vals = np.linalg.eig(C_i)
#     des_index = np.argsort(eig_vals)[::-1]
#     M_D_i = np.zeros((d_i, d))
#     for i in range(1, d_i + 1):
#         M_D_i[:, 1] = np.matrix.transpose(eig_vec[des_index[i]])
#     X_i = np.matmul(X_i_tilt, np.matrix.transpose(M_D_i))
#     return X_i


# def aglgp_prediction(X, y, sigma_e, corr_model, regions_nb, subclusters_nb, X_test):
#     model_dict = {'x': X, 'y': y, 'sigma_e': sigma_e, 'corr_model': corr_model,
#                   'regions_nb': regions_nb, 'subclusters_nb': subclusters_nb,
#                   'optimization_function': Optimization.optimization_scipy}
#     model = AGLGP.AGLGP(model_dict)
#     model.generate_model()
#     pred = model.predict(X_test)
#     return pred


# def sgpma(k, d, n, low_sizes, low_dims, nb_rep, nb_val, using_pca=True):
#     """
#
#     :param k:
#     :param d:
#     :param n:
#     :param low_sizes:
#     :param low_dims:
#     :param using_pca:
#     :return:
#     """
#     gen = generate_LHS(min_boundary=0, max_boundary=10, k=k, d=d, tests_nb=nb_val, k_test=10 * d)
#     X = gen['x']
#     X_test = gen['x_test'][0]
#
#     y = functions_average_experiments_2019(x=X, function_nb=1, random_perm=None)
#     y, sigma_e = replication_gen(y_det=y, nb_rep=nb_rep)
#
#     y_test = functions_average_experiments_2019(x=X_test, function_nb=1, random_perm=None)
#
#     if using_pca:
#         X_list, y_list = low_size_low_dim_subset_gen(X=X, Y=y, n=n, ki=low_sizes, di=low_dims, using_pca=True)
#         X_test_reduced_list = [PCA_reduction(X_test, di) for di in low_dims]
#         sigma_e_list = [noise_conf_2019(yi, noise_nb=1) for yi in y_list]
#
#     else:
#         X_list, y_list = low_size_low_dim_subset_gen(X, y, n, ki=low_sizes, di=low_dims, using_pca=False)
#         X_test_reduced_list = [random_embed(X_test, di) for di in low_dims]
#         sigma_e_list = [noise_conf_2019(yi, noise_nb=1) for yi in y_list]
#
#     # print(X_test_reduced_list[0].shape, len(X_test_reduced_list))
#     start = time.time()
#     pred_list = [mnek_prediction(X=X_list[i], y=y_list[i], sigma_e=sigma_e_list[i], regr_model=0, corr_model=1,
#                                  X_test=X_test_reduced_list[i])
#                  for i in range(n)]
#     runtime = time.time() - start
#
#     w = weights(k, d, low_sizes, low_dims, eta=8)
#     pred_w = np.sum([w[i] * pred_list[i] for i in range(n)], axis=0)
#
#     return {'weighted predictions': pred_w, 'MSE': MSE(y_test, pred_w), 'runtime': runtime}


# def sgpma_rep(k, d, n, low_sizes, low_dims, nb_rep, nb_val, type_of_reduction='pca'):
#     gen = generate_LHS(min_boundary=0, max_boundary=10, k=k, d=d, tests_nb=nb_val, k_test=10 * d)
#     X = gen['x']
#     X_test = gen['x_test']
#     y = f1(X)
#     # y_test = [f1(X_test[i]) for i in range(nb_val)]
#     # y_test = [stochastic_gen(f1(X_test[i]), nb_rep)['y'] for i in
#     #           range(nb_val)]  # the list of y_test corresponding to x_test for validation purposes
#     y_test = f1(X_test)
#     if type_of_reduction == 'random':
#         X_list, y_list = low_size_low_dim_subset_gen(X, y, n=n, ki=low_sizes, di=low_dims, emb='random')
#     else:
#         X_list, y_list = low_size_low_dim_subset_gen(X, y, n=n, ki=low_sizes, di=low_dims)
#     sigma_e = np.zeros((k, 1))
#     # y_list = []
#     # sigma_e_list = []
#     # for i in range(n):
#     #     sto_gen = stochastic_gen(y_det_list[i], nb_rep)
#     #     y_list.append(sto_gen['y'])
#     #     sigma_e_list.append(sto_gen['sigma_e'])
#     rmse = np.zeros((nb_val, 1))
#     mae = np.zeros((nb_val, 1))
#     r2 = np.zeros((nb_val, 1))
#     runtimes = np.zeros((nb_val, 1))
#     for j in range(nb_val):
#         if type_of_reduction == 'random':
#             X_test_reduced_list = [random_embed(X_test[j], di) for di in low_dims]
#         else:
#             X_test_reduced_list = [PCA_reduction(X_test[j], di) for di in low_dims]
#         start = time.time()
#         # pred_list = [mnek_prediction(X_list[i], y_list[i], sigma_e_list[i].reshape((sigma_e_list[i].shape[0], 1)),
#         #                              regr_model=0, corr_model=1, X_test=X_test_reduced_list[i]) for i in range(n)]
#         pred_list = [mnek_prediction(X_list[i], y_list[i], sigma_e,
#                                      regr_model=0, corr_model=1, X_test=X_test_reduced_list[i]) for i in range(n)]
#         runtime = time.time() - start
#         w = weights(k, d, low_sizes, low_dims, eta=5)
#         weighted_pred = np.sum([w[i] * pred_list[i] for i in range(n)], axis=0)
#         rmse[j, :] = mean_squared_error(y_test[j], weighted_pred, squared=False)
#         mae[j, :] = mean_absolute_error(y_test[j], weighted_pred)
#         r2[j, :] = r2_score(y_test[j], weighted_pred)
#         runtimes[j, :] = runtime
#     mean_accs = {'rmse': np.mean(rmse), 'mae': np.mean(mae), 'r2': np.mean(r2), 'runtime': np.mean(runtimes)}
#     macro_accs = {'rmses': rmse, 'maes': mae, 'r2s': r2, 'runtimes': runtimes}
#     return {'mean_accs': mean_accs, 'macro_accs': macro_accs}


def sgpma_predict(X, y, X_test, n_subsets, low_sizes, low_dims, regr_model, corr_model, eta, emb='pca'):
    num_of_samples, dim = X.shape
    if emb == 'random':
        X_list, y_list = low_size_low_dim_subset_gen(X, y, n_subsets, low_sizes, low_dims, emb='random')
        X_test_reduced_list = [random_embed(X_test, di) for di in low_dims]
    else:
        X_list, y_list = low_size_low_dim_subset_gen(X, y, n_subsets, low_sizes, low_dims)
        X_test_reduced_list = [PCA_reduction(X_test, di) for di in low_dims]
    start = time.time()
    model_list = [OK_Rmodel_kd_nugget(X_list[i], y_list[i], regr_model, corr_model, parameter_a=10) for i in range(n_subsets)]
    runtime = time.time() - start
    pred_list = [OK_Rpredict(model_list[i], X_test_reduced_list[i], regr_model)[0] for i in range(n_subsets)]
    w = weights(num_of_samples, dim, low_sizes, low_dims, eta)
    pred = np.sum([w[i] * pred_list[i] for i in range(n_subsets)], axis=0)
    result = {'pred': pred, 'runtime': runtime}
    return result

