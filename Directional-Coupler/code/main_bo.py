from helper import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_design', type=int, default=10 + 1, help='dimension of design variables')
    parser.add_argument('--N_initial', type=int, default=10, help='Number of initial simulations')
    parser.add_argument('--N_total', type=int, default=20, help='Number of total simulations')
    parser.add_argument('--eps', type=float, default=0.5, help='exploration space')
    parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
    parser.add_argument('--acq', type=int, default=0, help='0=lcb, 1=ei, 2=pi')
    parser.add_argument('--targ_power_ratio', type=float, default=0.5, help='target power split ratio (0.7 for 0.7:0.3)')
    parser.add_argument('--interp_kind', default='cubic', help='target power split ratio (0.7 for 0.7:0.3)')
    args = parser.parse_args()

    np.random.seed(args.seed)

    w0, initial_design_all, refer_x_value_all = get_init('./initial_design.mat', args.dim_design - 1)
    bound = np.concatenate([(w0 - args.eps * np.abs(w0)).reshape(-1, 1), (w0 + args.eps * np.abs(w0)).reshape(-1, 1)],
                           axis=1)
    obj_func = lambda x: customer_obj_func(x, refer_x_value_all, initial_design_all, args.targ_power_ratio, args.interp_kind)

    w0 = np.concatenate([w0, [0.02e-6]])  # make the height of ridge also a design variable
    bound = np.concatenate([bound, [[0.07e-9, 0.11e-6]]], axis=0)  # bound of the height

    print("-" * 10 + "Begin Bayes Opt" + "-" * 10 + "\n initial metrics: %f" % obj_func(w0))
    print("refer x value: ", refer_x_value_all)
    if args.acq == 0:
        acq = ['LCB1', 0.3]
    elif args.acq == 1:
        acq = ['EI']
    elif args.acq == 2:
        acq = ['PI']
    elif args.acq == 3:
        acq = ['LCB2', 1.0]
    elif args.acq == 4:
        acq = ['LCB3', 2.0]
    else:
        raise ValueError("acquisition function is not implemented")
    cur_best_w, cur_best_y = bayes_opt(obj_func, args.dim_design, args.N_total, args.N_initial, bound, acq,
                                       verbose=True, file_suffix=acq[0] + '_' + str(args.targ_power_ratio) + '_' + str(args.seed) + '_' + args.interp_kind)

    print("-" * 10 + "Finish Bayes Opt" + "-" * 10 + "\n optimal metrics: %f" % cur_best_y)

