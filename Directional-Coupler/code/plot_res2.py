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
    nodals2 = np.matmul(nodal, np.array([[1, 0], [0, -1]]))
    nodals3 = np.matmul(nodal, np.array([[-1, 0], [0, -1]]))
    nodals4 = np.matmul(nodal, np.array([[-1, 0], [0, 1]]))

    fig = plt.figure(dpi=720)
    ax = fig.add_subplot(111)
    polygon = Polygon(nodal)
    patch = PolygonPatch(polygon, fc=GRAY, ec=GRAY)
    ax.add_patch(patch)

    polygon = Polygon(nodals2)
    patch = PolygonPatch(polygon, fc=GRAY, ec=GRAY)
    ax.add_patch(patch)

    polygon = Polygon(nodals3)
    patch = PolygonPatch(polygon, fc=GRAY, ec=GRAY)
    ax.add_patch(patch)

    polygon = Polygon(nodals4)
    patch = PolygonPatch(polygon, fc=GRAY, ec=GRAY)
    ax.add_patch(patch)

    XLim, YLim = [-10, 10], [-2, 2]
    ax.set_xlim(XLim)
    ax.set_ylim(YLim)


def myplot(care_iter):
    Y = np.load("./result/Y_LCB1_0.npy")
    X = np.load("./result/X_LCB1_0.5_0_cubic.npy")
    # X = np.load("./result/X_LCB1_0.5_0_linear.npy")

    dim_design = 11

    _, initial_design_all, refer_x_value_all = get_init('./initial_design.mat', dim_design - 1)

    plot_with_nodal(initial_design_all)

    for i in range(len(care_iter)):
        cur_x, cur_y = X[:care_iter[i], :], Y[:care_iter[i]]
        index = np.argmin(cur_y)
        cur_optx = cur_x[index]
        print(cur_optx)
        data_fed_lumerical = data_tranform(cur_optx, refer_x_value_all,
                                           initial_design_all)

        plot_with_nodal(data_fed_lumerical)
        plt.show()


def normalize_points(nodals, refer=[0.2, 0.8]):
    num_modals, min_bound, max_bound = nodals.shape[0], nodals.min(axis=0), nodals.max(axis=0)

    normalized_nodals = (refer[1] - refer[0]) * np.divide(
        (nodals - min_bound.reshape(1, -1).repeat(num_modals, axis=0)),
        (max_bound - min_bound).reshape(1, -1).repeat(num_modals, axis=0)) + refer[0]
    return normalized_nodals


def myplot2():
    markersize = 10
    font_size = 28
    legend_size = 28
    weight = 'normal'
    ld = 3

    res = []
    with open('./sim_res.txt', 'r') as f:
        for line in f.readlines():
            cur_wavelength, cur_through, cur_cross = [float(ele) for ele in line.strip().split("\t")]
            res.append([cur_wavelength, cur_through, cur_cross])
    res = np.array(res)
    plt.figure()
    plt.plot(res[:, 0], res[:, 1], color='blue', linewidth=ld, label='through')
    plt.plot(res[:, 0], res[:, 2], color='orange', linewidth=ld, label='cross')
    plt.plot(res[:, 0], 0.629 * np.ones_like(res[:, 0]), '-.', color='green', linewidth=ld)
    plt.plot(res[:, 0], 0.397 * np.ones_like(res[:, 0]), '-.', color='green', linewidth=ld)
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
    myplot([120])
    # myplot2()
