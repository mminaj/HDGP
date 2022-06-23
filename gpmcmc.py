# -*- coding: utf-8 -*-

import numpy as np
import random
import math
import time
import warnings
import parmap

from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.linalg import eigh
from kriging_gpr.src.kriging_model.utils.normalize_data import *
from kriging_gpr.src.kriging_model.utils.OK_regr import *
from kriging_gpr.src.kriging_model.utils.OK_corr import *
from kriging_gpr.src.kriging_model.interface.OK_Rpredict import *


def lh_comp(comp, normal_data, theta, num_samples, corr_model, M):
    theta_comp = theta[M[comp], :]
    dim_comp = len(M[comp])
    normal_data_comp = normal_data[:, M[comp]]
    D_x_comp = np.zeros((dim_comp, num_samples, num_samples))
    for d in range(dim_comp):
        X = normal_data_comp[:, d].reshape(-1, 1) * np.ones((num_samples, num_samples))
        X_diff = np.zeros((num_samples, num_samples))
        for row in range(num_samples):
            x_row = X[row, row] * np.ones((1, num_samples - row))
            x_col = X[row:num_samples, row]
            X_row = x_row - x_col
            X_diff[row, row:num_samples] = X_row
            X_diff[row:num_samples, row] = -X_row
        D_x_comp[d, :, :] = X_diff.reshape((1, *X_diff.shape))
    corr_comp = OK_corr(corr_model, theta_comp, D_x_comp)
    return corr_comp


def model_comp(comp, normal_data, num_samples, corr_model, M):
    dim_comp = len(M[comp])
    normal_data_comp = normal_data[:, M[comp]]
    D_x_comp = np.zeros((dim_comp, num_samples, num_samples))
    temp_d_x_comp = np.zeros((num_samples * num_samples, dim_comp))
    for d in range(dim_comp):
        X = normal_data_comp[:, d].reshape(-1, 1) * np.ones((num_samples, num_samples))
        X_diff = np.zeros((num_samples, num_samples))
        for row in range(num_samples):
            x_row = X[row, row] * np.ones((1, num_samples - row))
            x_col = X[row:num_samples, row]
            X_row = x_row - x_col
            X_diff[row, row:num_samples] = X_row
            X_diff[row:num_samples, row] = -X_row
        D_x_comp[d, :, :] = X_diff.reshape((1, *X_diff.shape))
        temp_d_x_comp[:, d] = X_diff.flatten()
        theta_0_comp = (np.log(2) / dim_comp) * ((np.mean(np.abs(temp_d_x_comp), 0) + 1e-10) ** (-1 * corr_model))
        corr_comp = OK_corr(corr_model, theta_0_comp, D_x_comp)
    return M[comp], corr_comp, theta_0_comp, D_x_comp


def optimize_comp(comp, normal_data, data_out, regr_model, corr_model, theta_0, delta_lb, M):
    theta_0_comp = theta_0[M[comp],:]
    normal_data_comp = normal_data[:, M[comp]]
    regr_comp = OK_regr(normal_data_comp, regr_model)
    dim_comp = len(M[comp])
    lob_theta_comp = 0.000000001 * np.ones((dim_comp, 1))
    lower_bound_theta_comp = np.ndarray.flatten(lob_theta_comp)
    upper_bound_theta_comp = np.full(lower_bound_theta_comp.shape, np.inf)
    options = {'maxiter': 1000000}
    bnds_comp = Bounds(lower_bound_theta_comp, upper_bound_theta_comp)
    fun = lambda p_in: gp_mcmc_lh_comp(p_in, normal_data_comp, data_out, regr_comp, corr_model, delta_lb)

    class TookTooLong(Warning):
        pass
    class Minimizer:
        def __init__(self, timeout):
            self.timeout = timeout
        def minimize(self):
            self.start_time = time.time()
            res = minimize(fun, np.ndarray.flatten(theta_0_comp), method='Nelder-Mead', bounds=bnds_comp,
                                  options=options, callback = self.callback).x.reshape(-1, 1)
            return res
        def callback(self, x):
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout:
                warnings.warn("Terminating optimization: time limit reached, used {}s".format(elapsed),
                              TookTooLong)
                return True
    time_lim = 86 #172 # 24*60min/500iter = 2.88min/iter
    minimizer = Minimizer(time_lim)
    try:
        theta_comp = minimizer.minimize()
    except:
        theta_comp = theta_0_comp
    # theta_comp = minimize(fun, np.ndarray.flatten(theta_0_comp), method='Nelder-Mead', bounds=bnds_comp,
    #                       options=options).x.reshape(-1, 1)
    return M[comp], theta_comp


def gp_mcmc_lh_comp(params, data_in, Y, regr, corr_model, delta):
    num_samples, dim = data_in.shape
    normal_data, min_data, max_data = normalize_data(data_in)
    theta = params.reshape(-1, 1)
    D_x = np.zeros((dim, num_samples, num_samples))
    for d in range(dim):
        X = normal_data[:, d].reshape(-1, 1) * np.ones((num_samples, num_samples))
        X_diff = np.zeros((num_samples, num_samples))
        for row in range(num_samples):
            x_row = X[row, row] * np.ones((1, num_samples - row))
            x_col = X[row:num_samples, row]
            X_row = x_row - x_col
            X_diff[row, row:num_samples] = X_row
            X_diff[row:num_samples, row] = -X_row
        D_x[d, :, :] = X_diff.reshape((1, *X_diff.shape))
    corr = OK_corr(corr_model, theta, D_x)
    R = corr
    R = R + delta * (np.eye(R.shape[0], R.shape[1]))
    U = (np.linalg.cholesky(R)).transpose()
    L = np.transpose(U)
    Linv = np.linalg.inv(L)
    Sinv = Linv.transpose() @ Linv
    beta = np.linalg.inv(np.transpose(regr) @ Sinv @ regr) @ (np.transpose(regr) @ (Sinv @ Y))
    sigma_z = (1 / num_samples) * (np.transpose(Y - (regr @ beta)) @ Sinv @ (Y - (regr @ beta)))
    f = num_samples * (np.log(sigma_z)) + np.log(np.linalg.det(R))
    eps = 1e-15
    if np.isnan(f[0, 0]) or np.isinf(f[0, 0]):
        f = num_samples * (np.log(sigma_z + eps)) + np.log(np.linalg.det(R) + eps)
    return f[0, 0]


def gp_mcmc_lh(params, M, data_in, Y, regr, corr_model, delta):
    num_samples, dim = data_in.shape
    normal_data, min_data, max_data = normalize_data(data_in)
    P = len(M)
    theta = np.reshape(params, (dim, 1))
    corr_comp_list = parmap.map(lh_comp, range(P), normal_data, theta, num_samples, corr_model, M)
    corr = sum(corr_comp_list)
    R = corr
    R = R + delta * (np.eye(R.shape[0], R.shape[1]))
    U = (np.linalg.cholesky(R)).transpose()
    L = np.transpose(U)
    Linv = np.linalg.inv(L)
    Sinv = Linv.transpose() @ Linv
    beta = np.linalg.inv(np.transpose(regr) @ Sinv @ regr) @ (np.transpose(regr) @ (Sinv @ Y))
    sigma_z = (1 / num_samples) * (np.transpose(Y - (regr @ beta)) @ Sinv @ (Y - (regr @ beta)))
    f = num_samples * (np.log(sigma_z)) + np.log(np.linalg.det(R))
    eps = 1e-15
    if np.isnan(f[0, 0]) or np.isinf(f[0, 0]):
        f = num_samples * (np.log(sigma_z + eps)) + np.log(np.linalg.det(R) + eps)
    return f[0, 0]


def gp_mcmc_model(data_in, data_out, regr_model, corr_model, M, parameter_a=10):
    num_samples, dim = data_in.shape
    normal_data, min_data, max_data = normalize_data(data_in)
    regr = OK_regr(normal_data, regr_model)
    beta_0 = np.linalg.lstsq((np.transpose(regr) @ regr), (np.transpose(regr) @ data_out), rcond=None)
    beta_0 = beta_0[0]
    sigma_z0 = np.var(data_out - (regr @ beta_0))
    P = len(M)
    theta_0 = np.zeros((dim, 1))
    D_x = np.zeros((dim, num_samples, num_samples))
    corr = np.zeros((num_samples, num_samples))
    if corr_model == 0 or corr_model == 3:
        theta_0[:, 0] = 0.5
    else:
        list = parmap.map(model_comp, range(P), normal_data, num_samples, corr_model, M)
        for tup in list:
            corr = np.add(corr, tup[1])
            theta_0[tup[0], :] = tup[2].reshape(-1, 1)
            D_x[tup[0], :, :] = tup[3]
    a = parameter_a
    cond_ = np.linalg.cond(corr, p=2)
    exp_ = np.exp(a)
    if np.allclose(corr, corr.T, rtol=1e-7, atol=1e-10):
        eigen_v = eigh(corr, eigvals_only=True, subset_by_index=[corr.shape[0] - 1, corr.shape[0] - 1])[0]
    delta_lb = np.maximum(((eigen_v * (cond_ - exp_)) / (cond_ * (exp_ - 1))), 0)
    lob_sigma_z = 0.00001 * sigma_z0
    theta= np.zeros((dim, 1))
    list = parmap.map(optimize_comp, range(P), normal_data, data_out, regr_model, corr_model, theta_0, delta_lb, M)
    for tup in list:
        theta[tup[0], :] = tup[-1]
    model_evidence = gp_mcmc_lh(theta, M, normal_data, data_out, regr, corr_model, delta_lb)
    R = OK_corr(corr_model, theta, D_x)
    CR = (R + delta_lb * np.eye(R.shape[0], R.shape[1]))
    U0 = np.linalg.cholesky(CR).transpose()
    L = np.transpose(U0)
    D_L = np.transpose(U0)
    Linv = np.linalg.inv(L)
    Rinv = Linv.transpose() @ Linv
    beta = np.linalg.inv(np.transpose(regr) @ Rinv @ regr) @ (np.transpose(regr) @ (Rinv @ data_out))
    beta_v = np.linalg.inv(np.transpose(regr) @ Rinv @ regr) @ (np.transpose(regr) @ Rinv)
    sigma_z = (1 / num_samples) * (np.transpose(data_out - (regr @ beta)) @ Rinv @ (data_out - (regr @ beta)))
    M_model = {'sigma_z': sigma_z,
               'min_X': min_data,
               'max_X': max_data,
               'regr': regr,
               'beta': beta,
               'beta_v': beta_v,
               'theta': theta,
               'X': normal_data,
               'corr': corr_model,
               'L': L,
               'D_L': D_L,
               'Z': np.linalg.lstsq(L, (data_out - regr @ beta), rcond=None),
               'Z_v': np.linalg.lstsq(L, (np.eye(np.max(data_out.shape)) - regr @ beta_v), rcond=None),
               'Z_m': np.linalg.inv(L),
               'DZ_m': np.linalg.inv(D_L),
               'Rinv': Rinv,
               'nugget': delta_lb,
               'Y': data_out,
               'model_evidence': model_evidence}
    return M_model


def proposed_model(M):
    P = len(M)
    if P == 1:
        print('full model, can only split')
        comp_size = len(M[0]) - 1
        bar = math.ceil(comp_size * random.uniform(0, 1))
        half1 = M[0][:bar]
        half2 = M[0][bar:]
        del M[0]
        M.append(half1)
        M.append(half2)
    elif all(len(comp)==1 for comp in M):
        print('cannot split anymore, merge')
        comp_to_merge = random.sample(list(np.arange(P)),
                                      2)  # randomly choose 2 components uniformly from M without replacement
        comp1 = M[comp_to_merge[0]]
        comp2 = M[comp_to_merge[1]]
        M.remove(comp1)
        M.remove(comp2)
        merged_comp = comp1 + comp2
        M.append(merged_comp)
    else: # when at least 2 components with length NOT ALL = 1
        toss = random.uniform(0, 1)
        ids = [i for i, comp in enumerate(M) if len(comp) != 1] # filter out index of the components with length != 1
        if toss <= 0.5 :  # split
            print('split')
            comp_to_split = random.sample(ids, 1)[0]
            comp_size = len(M[comp_to_split]) - 1
            bar = math.ceil(comp_size * random.uniform(0, 1))
            half1 = M[comp_to_split][:bar]
            half2 = M[comp_to_split][bar:]
            del M[comp_to_split]
            M.append(half1)
            M.append(half2)
        else:  # merge
            print('merge')
            comp_to_merge = random.sample(list(np.arange(P)), 2)  # randomly choose 2 components uniformly from M without replacement
            comp1 = M[comp_to_merge[0]]
            comp2 = M[comp_to_merge[1]]
            M.remove(comp1)
            M.remove(comp2)
            merged_comp = comp1 + comp2
            M.append(merged_comp)
    Mp = M
    return Mp


def g(Mp, M):
    P = len(M)
    if P == 1:
        prob = 0
    elif len(M) <= len(Mp):
        prob = 0.5 * 1 / (2 ** (P - 1))
    else:
        prob = 0.5 * 1 / math.comb(P, 2)
    return prob


def metropolis_sampling(M0, data_in, data_out, regr_model, corr_model, n_iter):
    print('initial model', M0)
    Mj = M0
    M_list = []
    density = np.zeros((n_iter, 1))
    i = 0
    start = time.time()
    while i < n_iter and time.time() < start + 60*60*24:
        print('iteration {}...'.format(i))
        print('current', Mj)
        Mp = proposed_model(Mj)
        print('proposed', Mp)
        Mp_model = gp_mcmc_model(data_in, data_out, regr_model, corr_model, Mp)
        evi_Mp = Mp_model['model_evidence']
        Mj_model = gp_mcmc_model(data_in, data_out, regr_model, corr_model, Mj)
        evi_Mj = Mj_model['model_evidence']
        acpt_prob = min(1, (evi_Mp * g(Mj, Mp)) / (evi_Mj * g(Mp, Mj)))
        unif = random.uniform(0, 1)
        if acpt_prob > unif:
            print('accept proposed model!')
            Mj = Mp
        else: print('keep current model!')
        Mj_sorted = sorted([sorted(i) for i in Mj])
        M_list_sorted = [sorted([sorted(i) for i in comp]) for comp in M_list] # the order of component in M_list is not changed
        if not Mj_sorted in M_list_sorted or M_list == []:
            print('not found in list, append!')
            M_list.append(Mj.copy()) # only append Mj into model list if it did not exist previously
            ind_j = M_list.index(Mj.copy())
            density[ind_j, :] += 1
        else:
            print('already existed')
            match = [Mj_sorted == comp for comp in M_list_sorted]
            ind_j = [i for i, x in enumerate(match) if x] # returns the index of matched elements, ie the position of Mj in M_list
            density[ind_j, :] += 1 # counter list becomes the density of each Mj in n iterations
        i += 1
    # if i < k:
    #     k = i
    # k_models = random.sample(M_list, k)
    density = density[:len(M_list), :]
    result = {'M_list': M_list,
              'density': density}
                #,'k_models': k_models}
    return result


def gp_mcmc_predict(M_list, k, Xtest, data_in, data_out, regr_model, corr_model):
    if len(M_list) < k:
        k = len(M_list)
    k_models = random.sample(M_list, k)
    gp_model_list = [gp_mcmc_model(data_in, data_out, regr_model, corr_model, k_models[k]) for k in range(k)]
    preds = np.array([OK_Rpredict(gp_model_list[k], Xtest, regr_model)[0].flatten() for k in range(len(k_models))])
    pred = np.mean(preds, axis=0)
    result = {'pred': pred, 'k_models': k_models}
    return result


def get_full_model(dim):
    return [list(np.arange(dim))]


def get_init_model(dim):
    return [[d] for d in range(dim)]

