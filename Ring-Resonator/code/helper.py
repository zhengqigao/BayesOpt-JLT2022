import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np
import os
import time
from copy import deepcopy
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from scipy.optimize import LinearConstraint
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
import argparse
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.signal import chirp, find_peaks, peak_widths
import pickle


def perform_simulation(file_name='./zhengqi_ring_resonator_sim.lsf'):
    # prefix = "fdtd-solutions -run "
    # suffix = " -hide -exit"
    # os.system(prefix + file_name + suffix)
    prefix = "fdtd-solutions -trust-script -run "
    suffix = " -use-solve"
    os.system(prefix + file_name + suffix)
    return 1


def retrieve_res(file_name='./sim_res.txt'):
    res = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            cur_wavelength, p11, p12, p13, p14 = [float(ele) for ele in line.strip().split("\t")]
            res.append([cur_wavelength, p11, p12, p13, p14])
    os.remove(file_name)
    return np.array(res)


def simu_res_to_metrix(res):
    wavelength, p11, p12, p13, p14 = res[::-1, 0], res[::-1, 1], res[::-1, 2], res[::-1, 3], res[::-1, 4]

    target_wavelength = 1.55
    # p12 through, p14 drop
    target_ind = np.argmin(np.abs(wavelength - target_wavelength))

    indices, _ = find_peaks(p14, 0.1)

    if len(indices) == 0:  # no peak is found, a real bad design
        return 1000
    peak_ind = np.argmin(np.abs(wavelength[indices] - target_wavelength))  # the index of peak closest to 1.55

    peak_ind = indices[peak_ind]
    peak_power = p14[peak_ind]

    results_full = peak_widths(p14, [peak_ind], rel_height=0.5)

    f = interp1d(np.arange(len(wavelength)), wavelength, fill_value='extrapolation')
    lower_wl, upper_wl = f(results_full[2]), f(results_full[3])
    lower_wl, upper_wl = lower_wl[0], upper_wl[0]

    # plt.plot(wavelength, p14)
    # plt.plot(wavelength[indices], p14[indices], 'x')
    # plt.hlines(results_full[1],lower_wl,upper_wl,colors='C2')
    # plt.show()

    numerator = 3e8 / wavelength[peak_ind]
    denominator = 3e8 / lower_wl - 3e8 / upper_wl
    term1 = -numerator / denominator  # maximize Q factor
    term2 = (wavelength[peak_ind] - target_wavelength) ** 2  # oscilating freq close to target
    sigmoid = lambda x: np.exp(-np.logaddexp(0, -x))
    return (term1 + 1e7 * term2) * 1e-3


def data_tranform(design_vari):
    # several parameters might be revised
    num_dense = 50
    width_arc = 0.4e-6
    angles = np.linspace(-np.pi / 2.0, np.pi / 2.0, len(design_vari))
    f = interp1d(angles, design_vari, 'cubic')

    dense_angles = np.linspace(-np.pi / 2.0, np.pi / 2.0, num_dense)
    dense_radius = f(dense_angles)

    data_x, data_y = dense_radius * np.cos(dense_angles), dense_radius * np.sin(dense_angles)
    data_x_outer, data_y_outer = (dense_radius + width_arc) * np.cos(dense_angles), (dense_radius + width_arc) * np.sin(
        dense_angles)

    wrk1 = np.concatenate((data_x.reshape(-1, 1), data_y.reshape(-1, 1)), axis=1)
    wrk2 = np.concatenate((data_x_outer[::-1].reshape(-1, 1), data_y_outer[::-1].reshape(-1, 1)), axis=1)

    design_arc = np.concatenate((wrk1, wrk2), axis=0)

    scio.savemat('design_arc.mat',
                 {'design_arc': design_arc})
    return design_arc


def customer_obj_func(w):
    data_tranform(w)
    perform_simulation()
    wrk = retrieve_res()
    res = simu_res_to_metrix(wrk)
    return res


# def testfunc(x):
#     # return 100 * (x[1]-x[0])**2 + (1-x[0])**2
#     return 10 * x.shape[0] + np.sum(x * x - 10 * np.cos(2 * np.pi * x))
#     # return np.sin(2 * np.pi * np.sum(x))
#
#
# def test_gpr(func, N_initial, dim_design, w_bound):
#     # X_test = np.repeat((w_bound[:, 1] - w_bound[:, 0]).reshape(1, -1), N_initial, axis=0) * np.random.rand(
#     #    N_initial, dim_design) + np.repeat(w_bound[:, 0].reshape(1, -1), N_initial, axis=0)
#     # Y_test = [func(X_test[i, :]) for i in range(N_initial)]
#     X_test = np.load("./result/X_test.npy")
#     Y_test = np.load("./result/Y_test.npy")
#     print("finish generating test samples")
#     scale = 1e7
#     X_train, Y_train = np.load('./result/X_0.npy'), np.load('./result/Y_0.npy')
#     model = GPR(kernel=ConstantKernel(1.0, (1e-7, 1e7)) * RBF(0.1, (1e-2, 1e2)), normalize_y=True)
#     model.fit(X_train * scale, Y_train)
#     Y_hat = model.predict(X_test * scale)
#     print(Y_hat)
#     print(Y_test)
#     print(np.sum(np.abs(Y_hat - Y_test)) / len(Y_hat))
#     return 1
#
#
# def test_gpr2(dim_design, w_bound):
#     # X_test = np.repeat((w_bound[:, 1] - w_bound[:, 0]).reshape(1, -1), N_initial, axis=0) * np.random.rand(
#     #    N_initial, dim_design) + np.repeat(w_bound[:, 0].reshape(1, -1), N_initial, axis=0)
#     # Y_test = [func(X_test[i, :]) for i in range(N_initial)]
#
#     X_train, Y_train = np.load('./result/X_0.npy'), np.load('./result/Y_0.npy')
#     X_train, Y_train = X_train[:41, :], Y_train[:41]
#
#     # wrkmean, wrkstd = w_bound.mean(axis=1), (w_bound[:, 1] - w_bound[:, 0]) / 12 ** 0.5
#     wrkmean, wrkstd = X_train.mean(axis=0), X_train.std(axis=0)
#
#     model = GPR(kernel=ConstantKernel(1.0, (1e-7, 1e7)) * RBF(0.1, (1e-2, 1e2)), normalize_y=True)
#     model.fit(np.divide(X_train - wrkmean, wrkstd), Y_train)
#
#     # define and solve acquisition function
#     def acq_func(x_scaled):  # x_scaled: 1 * dim
#         x_scaled = x_scaled.reshape(1, -1)
#         mean, std = model.predict(x_scaled, return_std=True)
#         return mean[0] - 0.3 * std[0]
#
#     ntest = 1000
#     val_list = []
#     wrklist = np.zeros(ntest, )
#     for i in range(ntest):
#         wrk = ((w_bound[:, 1] - w_bound[:, 0]) * np.random.rand(dim_design) + (
#             w_bound[:, 0])).reshape(1, -1)
#         wrklist[i] = acq_func(np.divide(wrk - wrkmean, wrkstd))  # model.predict((wrk - wrkmean) / wrkstd)
#         val_list.append(wrk[0])
#     print(wrklist.mean(), wrklist.max(), wrklist.min())
#
#     w_init = (w_bound[:, 1] - w_bound[:, 0]) * np.random.rand(dim_design) + (
#         w_bound[:, 0])  # ((w_bound[:, 1] + w_bound[:, 0]) / 2.0)
#
#     # ind = np.argmin(wrklist)
#     # w_init = val_list[ind]
#
#     LC = LinearConstraint(np.eye(dim_design), np.divide(w_bound[:, 0] - wrkmean, wrkstd),
#                           np.divide(w_bound[:, 1] - wrkmean, wrkstd), keep_feasible=False)
#     opt = minimize(acq_func, np.divide(w_init - wrkmean, wrkstd), method='COBYLA', constraints=LC,
#                    options={'disp': False})
#
#     # wrkmean2 = np.repeat(wrkmean.reshape(-1, 1), 2, axis=1)
#     # wrkstd2 = np.repeat(wrkstd.reshape(-1, 1), 2, axis=1)
#     # opt = minimize(acq_func, np.divide(w_init - wrkmean, wrkstd), method='L-BFGS-B',
#     #                bounds=np.divide(w_bound - wrkmean2, wrkstd2),
#     #                options={'disp': True})
#
#     print(opt.x * wrkstd + wrkmean)
#     print(acq_func(opt.x))
#
#     return 1
#

# define and solve acquisition function
def acquisition(x_scaled, hyper_param, model, min_Y):  # x_scaled: 1 * dim
    x_scaled = x_scaled.reshape(1, -1)
    if 'LCB' in hyper_param[0]:
        mean, std = model.predict(x_scaled, return_std=True)
        return mean[0] - hyper_param[1] * std[0]
    elif hyper_param[0] == 'EI':
        tau = min_Y
        mean, std = model.predict(x_scaled, return_std=True)
        tau_scaled = (tau - mean) / std
        res = (tau - mean) * norm.cdf(tau_scaled) + std * norm.pdf(tau_scaled)
        return -res  # maximize Ei = minimize -EI
    elif hyper_param[0] == 'PI':
        tau = min_Y
        mean, std = model.predict(x_scaled, return_std=True)
        tau_scaled = (tau - mean) / std
        res = norm.cdf(tau_scaled)
        return -res
    else:
        raise ValueError("acquisition function is not implemented")


def bayes_opt(func, dim_design, N_sim, N_initial, w_bound, hyper_param, verbose=True, file_suffix=''):
    # initialization
    print("Begin initializing...")
    X = np.repeat((w_bound[:, 1] - w_bound[:, 0]).reshape(1, -1), N_initial, axis=0) * np.random.rand(
        N_initial, dim_design) + np.repeat(w_bound[:, 0].reshape(1, -1), N_initial, axis=0)
    Y = np.zeros((N_initial,))

    pred_mean = np.zeros(N_sim - N_initial)
    pred_std = np.zeros(N_sim - N_initial)
    acq_list = np.zeros(N_sim - N_initial)

    for i in range(N_initial):
        Y[i] = func(X[i, :])
        print("Simulate the %d-th sample... with metric: %f" % (i, Y[i])) if verbose else None
    print("Finish initialization with best metric: %f" % (np.min(Y)))

    # goes into bayesian optimization
    cur_count, cur_best_w, cur_best_y = N_initial, None, 1e10
    while cur_count < N_sim:

        wrk_mean, wrk_std = X.mean(axis=0), X.std(axis=0)

        # build gaussian process
        model = GPR(kernel=ConstantKernel(1.0, (1e-7, 1e7)) * RBF(0.1, (1e-2, 1e2)), normalize_y=True,
                    n_restarts_optimizer=10)
        model.fit(np.divide(X - wrk_mean, wrk_std), Y)

        acq_func = lambda x_scaled: acquisition(x_scaled, hyper_param, model, np.min(Y))

        # multi start point optimization
        cur_min, opt = 1e11, None
        for i in range(20):
            w_init = (w_bound[:, 1] - w_bound[:, 0]) * np.random.rand(dim_design) + (
                w_bound[:, 0])  # ((w_bound[:, 1] + w_bound[:, 0]) / 2.0)

            LC = LinearConstraint(np.eye(dim_design), np.divide(w_bound[:, 0] - wrk_mean, wrk_std),
                                  np.divide(w_bound[:, 1] - wrk_mean, wrk_std), keep_feasible=False)
            cur_opt = minimize(acq_func, np.divide(w_init - wrk_mean, wrk_std), method='COBYLA', constraints=LC,
                           options={'disp': False})
            wrk = acq_func(cur_opt.x)
            if cur_min >= wrk:
                cur_min = wrk
                opt = cur_opt

        # opt = minimize(acq_func, w_init * scale, method='L-BFGS-B', bounds=w_bound * scale, options={'disp': True, 'eps': 1e-3, 'gtol': 1e-07})

        # do a clipping to avoid violation of constraints
        newX = np.clip(opt.x * wrk_std + wrk_mean, w_bound[:, 0], w_bound[:, 1])
        star_time = time.time()
        cur_count += 1
        newY = func(newX)
        end_time = time.time()

        X = np.concatenate((X, newX.reshape(1, -1)), axis=0)
        Y = np.concatenate((Y, [newY]), axis=0)

        # save and display information
        ind = np.argmin(Y)
        cur_predmean, cur_predstd = model.predict(
            (np.divide(newX - wrk_mean, wrk_std)).reshape(1, -1), return_std=True)
        cur_acq = acq_func(np.divide(newX - wrk_mean, wrk_std))
        cur_best_w, cur_best_y = X[ind, :], Y[ind]
        pred_mean[cur_count - N_initial - 1], pred_std[cur_count - N_initial - 1] = cur_predmean, cur_predstd
        acq_list[cur_count - N_initial - 1] = cur_acq
        np.save('./result/X_' + file_suffix + '.npy', X)
        np.save('./result/Y_' + file_suffix + '.npy', Y)
        np.save('./result/cur_best_w_' + file_suffix + '.npy', cur_best_w)
        np.save('./result/cur_best_y_' + file_suffix + '.npy', cur_best_y)
        np.save('./result/pred_mean_' + file_suffix + '.npy', pred_mean)
        np.save('./result/pred_std_' + file_suffix + '.npy', pred_std)
        np.save('./result/acq_list_' + file_suffix + '.npy', acq_list)
        with open('./result/gpmodel_' + file_suffix + '.pkl','wb') as f:
            pickle.dump(model, f)

        if verbose:
            print("-" * 10)
            print("Number of function evaluations: %d" % cur_count)
            print("Optimize acq message: ", opt.message)
            print("Optimize initial X - new sampled X: ", w_init - newX)
            print("Model predict(new sampled X)... mean: %f, std:%f" % (cur_predmean, cur_predstd))
            print("Acq(new sampled X): %f" % cur_acq)
            print("Y(new sampled X): %f, simulation time: %f" % (newY, end_time - star_time))
            print("Current best design: ", cur_best_w)
            print("Current best function value: %f" % cur_best_y)

            ntest = 1000
            wrklist = np.zeros(ntest, )
            for i in range(X.shape[0]):
                wrklist[i] = model.predict((np.divide(X[i, :] - wrk_mean, wrk_std).reshape(1, -1)))
            for i in range(X.shape[0], ntest):
                wrk = (w_bound[:, 1] - w_bound[:, 0]) * np.random.rand(dim_design) + (
                    w_bound[:, 0])
                wrklist[i] = model.predict((np.divide(wrk - wrk_mean, wrk_std)).reshape(1, -1))
            print("Gpr model mean:%.3e, max: %.3e, min: %.3e" % (wrklist.mean(), wrklist.max(), wrklist.min()))

    # sanity check
    lb_x = X - w_bound[:, 0].reshape(1, -1).repeat(X.shape[0], axis=0)
    ub_x = X - w_bound[:, 1].reshape(1, -1).repeat(X.shape[0], axis=0)
    if np.any(lb_x < 0) or np.any(ub_x > 0):
        print("constraint of design variables is violated due to numerical issues. Check the value of X")
    return cur_best_w, cur_best_y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N_initial', type=int, default=5, help='Number of initial simulations')
    parser.add_argument('--N_total', type=int, default=20, help='Number of total simulations')
    parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
    parser.add_argument('--acq', type=int, default=0, help='0,3,4=lcb, 1=ei, 2=pi')
    args = parser.parse_args()

    np.random.seed(args.seed)
    bound = np.array([[0, 3.0], [0, 0.2]])
    dim_design = 2
    # w0 = np.concatenate([3.4e-6 * np.ones((dim_design // 2, 1)), 3.0e-6 * np.ones((dim_design // 2, 1))], axis=0).flatten()
    # w0 = np.array([3.4,3.4,3.4,3.35,3.38,3.35,3.4,3.4,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0])*1e-6

    w0 = np.array([1.6, 0.1])
    bayes_opt(lambda x: x, dim_design, args.N_total, args.N_initial, bound, ['LCB1', 0.3],
              verbose=True, file_suffix='LCB1' + '_' + str(args.seed))
