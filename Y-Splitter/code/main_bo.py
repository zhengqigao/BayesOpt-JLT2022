from helper import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N_initial', type=int, default=40, help='Number of initial simulations')
    parser.add_argument('--N_total', type=int, default=120, help='Number of total simulations')
    parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
    parser.add_argument('--acq', type=int, default=0, help='0,3,4=lcb, 1=ei, 2=pi')
    args = parser.parse_args()

    np.random.seed(args.seed)

    w0 = 0.6e-6 * np.ones(13, ) # square initialization
    dim_design = len(w0)

    
    bound = np.concatenate([(0.25e-6 * np.ones_like(w0)).reshape(-1, 1), (0.8e-6 * np.ones_like(w0)).reshape(-1, 1)],
                           axis=1)
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
    else:
        raise ValueError("acquisition function is not implemented")

    cur_best_w, cur_best_y = bayes_opt(obj_func, dim_design, args.N_total, args.N_initial, bound, acq,
                                       verbose=True, file_suffix=acq[0] + '_' + str(args.seed))

    # check values
    data_tranform(cur_best_w)
    print("-" * 10 + "Finish Bayes Opt" + "-" * 10 + "\n optimal metrics: %f" % cur_best_y)

    # cur_best_w = np.load('./result/cur_best_w_' + str(args.seed) + '.npy')
    # data_tranform(cur_best_w)  # generate a design_vari.mat in the current folder based on the obtained result.
