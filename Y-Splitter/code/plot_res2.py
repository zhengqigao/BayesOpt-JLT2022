import scipy.interpolate
from matplotlib import pyplot as plt
import numpy as np
import scipy.io as scio
from helper import *
from shapely.geometry import Polygon, Point
from descartes.patch import PolygonPatch
from matplotlib import pyplot as plt
import matplotlib

plt.rcParams['axes.linewidth'] = 3
font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 28}

matplotlib.rc('font', **font)


def plot_with_nodal(nodal):
    nodal = nodal * 1e6
    GRAY = '#999999'
    # nodals2 = np.matmul(nodal, np.array([[1, 0], [0, -1]]))
    # nodals3 = np.matmul(nodal, np.array([[-1, 0], [0, -1]]))
    # nodals4 = np.matmul(nodal, np.array([[-1, 0], [0, 1]]))

    fig = plt.figure(dpi=720)
    ax = fig.add_subplot(111)
    polygon = Polygon(nodal)
    patch = PolygonPatch(polygon, fc=GRAY, ec=GRAY)
    ax.add_patch(patch)

    polygon = Polygon(nodal)
    patch = PolygonPatch(polygon, fc=GRAY, ec=GRAY)
    ax.add_patch(patch)

    XLim, YLim = [-1, 3], [-1, 1]
    ax.set_xlim(XLim)
    ax.set_ylim(YLim)


def helper_func(w):
    x = np.linspace(0, 2, 101) * 1e-6
    xs = np.linspace(0, 2, 13) * 1e-6

    cs1 = scipy.interpolate.CubicSpline(xs, w)
    cs2 = scipy.interpolate.CubicSpline(xs, -w)
    yupper = cs1(x)
    ylower = cs2(x)
    nodals1 = np.concatenate((x.reshape(-1, 1), yupper.reshape(-1, 1)), axis=1)
    nodals2 = np.concatenate((x[-1::-1].reshape(-1, 1), ylower[-1::-1].reshape(-1, 1)), axis=1)
    nodal = np.concatenate((nodals1, nodals2), axis=0)
    print(nodal)
    return nodal


def myplot(care_iter):
    Y = np.load("./result/Y_LCB1_0.npy")
    X = np.load("./result/X_LCB1_0.npy")

    dim_design = 13

    w0 = 0.6e-6 * np.ones(13, )  # square initialization
    tmp = helper_func(w0)

    plot_with_nodal(tmp)

    for i in range(len(care_iter)):
        cur_x, cur_y = X[:care_iter[i], :], Y[:care_iter[i]]
        index = np.argmin(cur_y)
        cur_optx = cur_x[index]
        print(cur_optx)
        print(cur_y[index])
        tmp2 = data_tranform(cur_optx)
        tmp3 = helper_func(tmp2)
        plot_with_nodal(tmp3)
        plt.show()


def normalize_points(nodals, refer=[0.2, 0.8]):
    num_modals, min_bound, max_bound = nodals.shape[0], nodals.min(axis=0), nodals.max(axis=0)

    normalized_nodals = (refer[1] - refer[0]) * np.divide(
        (nodals - min_bound.reshape(1, -1).repeat(num_modals, axis=0)),
        (max_bound - min_bound).reshape(1, -1).repeat(num_modals, axis=0)) + refer[0]
    return normalized_nodals


def myplot2():
    markersize = 10
    font_size = 24
    legend_size = 24
    weight = 'normal'
    ld = 3

    res = []
    with open('./sim_res.txt', 'r') as f:
        for line in f.readlines():
            cur_wavelength, cur_through, cur_cross, _ = [float(ele) for ele in line.strip().split("\t")]
            res.append([cur_wavelength, cur_through, cur_cross])
    res = np.array(res)
    plt.figure()
    plt.plot(res[:, 0], res[:, 1], color='blue', linewidth=ld, label=r'$p^{12}$')
    plt.plot(res[:, 0], res[:, 2], color='orange', linewidth=ld, label=r'$p^{11}$')
    print("mean:%f, std:%f" % (np.mean(res[:, 1]), np.std(res[:, 1])))
    print("mean:%f, std:%f" % (np.mean(res[:, 2]), np.std(res[:, 2])))
    # plt.plot(res[:, 0], 0.629 * np.ones_like(res[:, 0]), '-.', color='green', linewidth=ld)
    # plt.plot(res[:, 0], 0.397 * np.ones_like(res[:, 0]), '-.', color='green', linewidth=ld)
    plt.ylabel('Normalized power', fontdict={'family': 'Times New Roman', 'size': font_size, 'weight': weight})
    plt.xlabel('Wavelength',
               fontdict={'family': 'Times New Roman', 'size': font_size, 'weight': weight})
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(ld)
    ax.spines['left'].set_linewidth(ld)
    ax.spines['right'].set_linewidth(ld)
    ax.spines['top'].set_linewidth(ld)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], font='Times New Roman', size=font_size)
    plt.xticks([1.50, 1.55, 1.60], font='Times New Roman', size=font_size)
    plt.legend(prop={'family': 'Times New Roman', 'size': legend_size, 'weight': weight})

    plt.show()


if __name__ == '__main__':
    #     care_iter = [100, 200]
    myplot([69])
    # myplot2()
    # wrk = np.load("./result/cur_best_y_ga_0.npy")
    # print(wrk)
    # Y = np.load("./result/cur_best_y_LBFGS_0.npy")
    # X = np.load("./result/cur_best_w_LBFGS_0.npy")
    # print(X)
    # print(Y)
    # dim_design = 13
    # tmp2 = data_tranform(X)
    # tmp3 = helper_func(tmp2)
    # plot_with_nodal(tmp3)
    # plt.show()
