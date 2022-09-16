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
from scipy.interpolate import interp1d
from scipy.stats import norm
import pickle

def perform_simulation(file_name='./DC_FDTD.lsf'):
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
            cur_wavelength, cur_through, cur_cross = [float(ele) for ele in line.strip().split("\t")]
            res.append([cur_wavelength, cur_through, cur_cross])
    os.remove(file_name)
    return np.array(res)


def simu_res_to_metrix(res, target_power_ratio):
    through = target_power_ratio
    cross = 1 - through
    return (np.sum((res[:, 1] - through) * (res[:, 1] - through)) + np.sum((res[:, 2] - cross) * (res[:, 2] - cross))) / res.shape[
        0] / 2.0
    # return (np.sum((res[:, 1] - res[:, 2]) * (res[:, 1] - res[:, 2])) + np.sum(
    #     (1 - res[:, 1] - res[:, 2]) * (1 - res[:, 1] - res[:, 2]))) / res.shape[0]


def extract_points(initial_design):
    # the initial_design we retrieved from lumerical simulation script is the right-upper arm.
    # 65     66    ----->  129
    # 64     63 <-----1 <- 130
    # values of indices=63, 64, 65, 66, 129, 130 are all fixed, [1,62] and [67,128] are design variables.
    # Namely, upper side has 62 variables, lower side has 62 variables.
    initial_design_lower, initial_design_upper = initial_design[0 + 5:40 - 5, :], initial_design[168 + 5:208 - 5]

    return initial_design_lower, initial_design_upper


def linear_projection(dim_design, x_value):
    # x_value (already sorted) defines the coordinates of all x in the initial designs
    # dim_design is used in Bayesian Optimization. An interpolation is needed
    wrk = np.arange(0, dim_design)
    k = (x_value[0] - x_value[-1]) / (wrk[0] - wrk[-1])
    refer_x_value = k * (wrk - wrk[0]) + x_value[0]
    return refer_x_value


def data_tranform(design_vari, refer_x_value_all, initial_design_all, interp_kind='cubic'):
    # design_vari: design variable, dim: (N, )
    # the given design variable is for one arm in the directional coupler.
    # Since we have four branches and considering symmetry, the coordinates are simply mirrored.
    # Also we need interpolate first

    scio.savemat('design_vari2.mat', {'design_vari2': design_vari[-1]})  # save the height of ridge
    design_vari = design_vari[:-1]

    dim_design = design_vari.shape[0]

    data_fed_lumerical = deepcopy(initial_design_all)

    # the order of indices should guruantee that x is increasing (required by numpy)
    # data_fed_lumerical[39::-1, 1] = np.interp(initial_design_all[39::-1, 0], refer_x_value_all[dim_design // 2 - 1::-1],
    #                                           design_vari[dim_design // 2 - 1::-1])
    # data_fed_lumerical[168:208, 1] = np.interp(initial_design_all[168:208, 0], refer_x_value_all[dim_design // 2:],
    #                                            design_vari[dim_design // 2:])

    wrk1 = np.concatenate(([initial_design_all[39, 0]], refer_x_value_all[dim_design // 2 - 1::-1], [initial_design_all[0, 0]]))
    wrk2 = np.concatenate(([initial_design_all[39, 1]], design_vari[dim_design // 2 - 1::-1], [initial_design_all[0, 1]]))
    f1 = interp1d(wrk1, wrk2, kind=interp_kind, fill_value="extrapolate")
    data_fed_lumerical[39::-1, 1] = f1(initial_design_all[39::-1, 0])

    wrk1 = np.concatenate(([initial_design_all[168, 0]], refer_x_value_all[dim_design // 2:], [initial_design_all[207, 0]]))
    wrk2 = np.concatenate(([initial_design_all[168, 1]], design_vari[dim_design // 2:], [initial_design_all[207, 1]]))
    f1 = interp1d(wrk1, wrk2, kind=interp_kind, fill_value="extrapolate")
    data_fed_lumerical[168:208, 1] = f1(initial_design_all[168:208, 0])

    scio.savemat('design_vari.mat',
                 {'design_vari': data_fed_lumerical})  # this mat file will be imported in lumerical script

    return data_fed_lumerical


def get_init(file_name, dim_design):
    initial_design_all = scio.loadmat(file_name)['init_design']  # load the initial design
    initial_design_lower, initial_design_upper = extract_points(initial_design_all)

    refer_x_value_lower = linear_projection(dim_design // 2, initial_design_lower[:, 0])
    refer_x_value_upper = linear_projection(dim_design // 2, initial_design_upper[:, 0])
    refer_x_value_all = np.concatenate((refer_x_value_lower, refer_x_value_upper), axis=0)

    # get initial value of w. Note that np.interp requires an increasing input
    w = np.zeros((dim_design,))
    w[dim_design // 2 - 1::-1] = np.interp(refer_x_value_lower[::-1], initial_design_lower[::-1, 0],
                                           initial_design_lower[::-1, 1])
    w[dim_design // 2:] = np.interp(refer_x_value_upper, initial_design_upper[:, 0], initial_design_upper[:, 1])

    return w, initial_design_all, refer_x_value_all


def customer_obj_func(w, refer_x_value_all, initial_design_all, target_power_ratio, interp_kind='cubic'):
    data_tranform(w, refer_x_value_all, initial_design_all, interp_kind)
    perform_simulation()
    res = retrieve_res()
    return simu_res_to_metrix(res, target_power_ratio)


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
        tau_scaled = (tau - mean)/std
        res = (tau - mean) * norm.cdf(tau_scaled) + std * norm.pdf(tau_scaled)
        return -res # maximize Ei = minimize -EI
    elif hyper_param[0] == 'PI':
        tau = min_Y
        mean, std = model.predict(x_scaled, return_std=True)
        tau_scaled = (tau - mean)/std
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

        acq_func = lambda x_scaled: acquisition(x_scaled, hyper_param, model, np.min(Y)) # optimal Y value is needed in EI and PI

        w_init = (w_bound[:, 1] - w_bound[:, 0]) * np.random.rand(dim_design) + (
            w_bound[:, 0])  # ((w_bound[:, 1] + w_bound[:, 0]) / 2.0)

        LC = LinearConstraint(np.eye(dim_design), np.divide(w_bound[:, 0] - wrk_mean, wrk_std),
                              np.divide(w_bound[:, 1] - wrk_mean, wrk_std), keep_feasible=False)
        opt = minimize(acq_func, np.divide(w_init - wrk_mean, wrk_std), method='COBYLA', constraints=LC,
                       options={'disp': False})

        # opt = minimize(acq_func, w_init * scale, method='L-BFGS-B', bounds=w_bound * scale, options={'disp': True, 'eps': 1e-3, 'gtol': 1e-07})

        # do a clipping to avoid violation of constraints
        newX = np.clip(opt.x * wrk_std + wrk_mean, w_bound[:, 0], w_bound[:, 1])
        star_time = time.time()
        cur_count += 1
        newY = func(newX)
        end_time = time.time()
        X, Y = np.concatenate((X, newX.reshape(1, -1)), axis=0), np.concatenate((Y, [newY]), axis=0)

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
        with open('./result/gpmodel_' + file_suffix + '.pkl', 'wb') as f:
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


# def test():
#     bound, hyper = np.array([[-5.12, 5.12], [-5.12, 5.12]]), ['LCB', 1.0]
#     best_w, best_y = bayes_opt(testfunc, 2, 1000, 100, bound, hyper)
#     return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_design', type=int, default=10 + 1, help='dimension of design variaables')
    parser.add_argument('--N_initial', type=int, default=40, help='Number of initial simulations')
    parser.add_argument('--N_total', type=int, default=200, help='Number of total simulations')
    parser.add_argument('--eps', type=float, default=0.7, help='exploration space')
    parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
    args = parser.parse_args()

    np.random.seed(args.seed)

    w0, initial_design_all, refer_x_value_all = get_init('./initial_design.mat', args.dim_design - 1)
    bound = np.concatenate([(w0 - args.eps * np.abs(w0)).reshape(-1, 1), (w0 + args.eps * np.abs(w0)).reshape(-1, 1)],
                           axis=1)
    obj_func = lambda x: customer_obj_func(x, refer_x_value_all, initial_design_all)

    w0 = np.concatenate([w0, [0.02e-6]])  # make the height of ridge also a design variable
    bound = np.concatenate([bound, [[0.07e-9, 0.11e-6]]], axis=0)  # bound of the height
    #
    cur_best_w, cur_best_y = bayes_opt(obj_func, args.dim_design, args.N_total, args.N_initial, bound, ['LCB', 0.3],
                                       verbose=True, file_suffix=str(args.seed))

    # test_gpr2(args.dim_design, bound)
