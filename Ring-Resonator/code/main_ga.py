from helper import *
import argparse
from scipy.optimize import differential_evolution
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N_iter', type=int, default=1, help='Number of total simulations')
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

    opt = differential_evolution(obj_func, bound, maxiter=args.N_iter, disp=True, popsize=3, init='random',
                                 mutation=1.5, polish=False)

    file_suffix = 'ga' + '_' + str(args.seed)
    np.save('./result/cur_best_w_' + file_suffix + '.npy', opt.x)
    np.save('./result/cur_best_y_' + file_suffix + '.npy', opt.fun)
    print("function evaluation %d" %opt.nfev)