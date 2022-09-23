import numpy as np

from helper import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N_initial', type=int, default=10, help='Number of initial simulations')
    parser.add_argument('--N_total', type=int, default=50, help='Number of total simulations')
    parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
    parser.add_argument('--acq', type=int, default=0, help='0,3,4=lcb, 1=ei, 2=pi')
    args = parser.parse_args()

    np.random.seed(args.seed)

    dim_design = 10

    # at least an oscilating solution
    # w0 = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3]) * 1e-6

    w0 = np.ones(dim_design) * 2.95e-6
    data_tranform(w0)

    bound = np.concatenate([2.9e-6 * np.ones_like(w0).reshape(-1, 1), 3.0e-6 * np.ones_like(w0).reshape(-1, 1)], axis=1)

    obj_func = lambda x: customer_obj_func(x)

    print("-" * 10 + "Begin Bayes Opt" + "-" * 10 + "\n initial metrics: %f" % obj_func(w0))

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
    elif args.acq == 5:
        acq = ['LCB4', 0.1]
    elif args.acq == 6:
        acq = ['LCB5', 0.5]
    elif args.acq == 7:
        acq = ['LCB6', 0.8]
    else:
        raise ValueError("acquisition function is not implemented")

    cur_best_w, cur_best_y = bayes_opt(obj_func, dim_design, args.N_total, args.N_initial, bound, acq,
                                       verbose=True, file_suffix=acq[0] + '_' + str(args.seed))

    # check values
    data_tranform(cur_best_w)
    print("-" * 10 + "Finish Bayes Opt" + "-" * 10 + "\n optimal metrics: %f" % cur_best_y)

    # # # cur_best_w = np.load('./result/cur_best_w_' + str(args.seed) + '.npy')
    # # # data_tranform(cur_best_w)  # generate a design_vari.mat in the current folder based on the obtained result.
