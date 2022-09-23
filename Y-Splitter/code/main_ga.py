from helper import *
import argparse
from scipy.optimize import differential_evolution
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N_iter', type=int, default=1, help='Number of total simulations')
    parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
    parser.add_argument('--acq', type=int, default=0, help='0,3,4=lcb, 1=ei, 2=pi')
    args = parser.parse_args()

    np.random.seed(args.seed)

    
    w0 = 0.6e-6 * np.ones(13, )  # square initialization for the junction part
    dim_design = len(w0)

    
    bound = np.concatenate([(0.25e-6 * np.ones_like(w0)).reshape(-1, 1), (0.8e-6 * np.ones_like(w0)).reshape(-1, 1)],
                           axis=1)
    obj_func = lambda x: customer_obj_func(x)

    opt = differential_evolution(obj_func, bound, maxiter=args.N_iter, disp=True, popsize=4)
    file_suffix = 'ga' + '_' + str(args.seed)
    np.save('./result/cur_best_w_' + file_suffix + '.npy', opt.x)
    np.save('./result/cur_best_y_' + file_suffix + '.npy', opt.fun)
    print("function evaluation %d" %opt.nfev)