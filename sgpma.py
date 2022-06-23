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


def weights(k, d, ki, di, eta):
    pi = np.multiply((ki / k) ** 2, (di / d) ** eta)
    return pi * 1 / np.sum(pi, axis=0)


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

