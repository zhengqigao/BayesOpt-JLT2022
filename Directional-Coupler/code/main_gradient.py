from helper import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_design', type=int, default=10 + 1, help='dimension of design variables')
    parser.add_argument('--N_total', type=int, default=20, help='Number of total simulations')
    parser.add_argument('--eps', type=float, default=0.5, help='exploration space')
    parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
    parser.add_argument('--targ_power_ratio', type=float, default=0.5, help='target power split ratio (0.7 for 0.7:0.3)')
    args = parser.parse_args()

    np.random.seed(args.seed)

    w0, initial_design_all, refer_x_value_all = get_init('./initial_design.mat', args.dim_design - 1)
    bound = np.concatenate([(w0 - args.eps * np.abs(w0)).reshape(-1, 1), (w0 + args.eps * np.abs(w0)).reshape(-1, 1)],
                           axis=1)
    obj_func = lambda x: customer_obj_func(x, refer_x_value_all, initial_design_all, args.targ_power_ratio)

    w0 = np.concatenate([w0, [0.02e-6]])  # make the height of ridge also a design variable
    bound = np.concatenate([bound, [[0.07e-9, 0.11e-6]]], axis=0)  # bound of the height

    opt = minimize(obj_func, w0, method='L-BFGS-B', bounds=bound, options={'disp': 99, 'maxfun': args.N_total, 'gtol':1e-9})
    file_suffix = 'LBFGS' + '_' + str(args.targ_power_ratio) + '_' + str(args.seed)
    np.save('./result/cur_best_w_' + file_suffix + '.npy', opt.x)
    np.save('./result/cur_best_y_' + file_suffix + '.npy', opt.fun)
    print("function evaluation %d" %opt.nfev)