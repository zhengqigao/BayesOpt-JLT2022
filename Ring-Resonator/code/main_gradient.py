from helper import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N_total', type=int, default=3, help='Number of total simulations')
    parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
    args = parser.parse_args()

    np.random.seed(args.seed)

    dim_design = 10

    # at least an oscilating solution
    # w0 = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3]) * 1e-6

    w0 = np.ones(dim_design) * 2.95e-6
    data_tranform(w0)

    bound = np.concatenate([2.9e-6 * np.ones_like(w0).reshape(-1, 1), 3.0e-6 * np.ones_like(w0).reshape(-1, 1)], axis=1)

    obj_func = lambda x: customer_obj_func(x)

    opt = minimize(obj_func, w0, method='L-BFGS-B', bounds=bound, options={'disp': 99, 'maxfun': args.N_total, 'gtol':1e-9})
    file_suffix = 'LBFGS' + '_' + str(args.seed)
    np.save('./result/cur_best_w_' + file_suffix + '.npy', opt.x)
    np.save('./result/cur_best_y_' + file_suffix + '.npy', opt.fun)
    print("function evaluation %d" %opt.nfev)