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


    XLim, YLim = [-5, 3], [-4, 4]
    ax.set_xlim(XLim)
    ax.set_ylim(YLim)



def myplot(care_iter):
    Y = np.load("./result/Y_EI_0.npy")
    X = np.load("./result/X_EI_0.npy")

    dim_design = 10

    w0 = np.ones(dim_design) * 2.95e-6
    tmp = data_tranform(w0)
    tmp[:,0] = -tmp[:,0]
    plot_with_nodal(tmp)
    print(Y[0])
    evolve = [w0,]
    for i in range(len(care_iter)):
        cur_x, cur_y = X[:care_iter[i], :], Y[:care_iter[i]]
        index = np.argmin(cur_y)
        cur_optx = cur_x[index]
        print(cur_optx)
        print(cur_y[index])
        evolve.append(cur_optx)
        tmp2 = data_tranform(cur_optx)
        tmp2[:,0] = -tmp2[:,0]
        plot_with_nodal(tmp2)
        plt.show()

    font_size = 26
    weight = 'normal'
    color_list = ['orange','blue','green']
    plt.figure()
    for i in range(len(evolve)):
        plt.plot(np.arange(len(evolve[0])), 1e6 * evolve[i], linewidth=3,color=color_list[i])
    plt.yticks([2.9, 3.0], font='Times New Roman', size=font_size)
    plt.xticks([0,2,4,6,8,10], font='Times New Roman', size=font_size)
    plt.ylabel('Radius value (um)', fontdict={'family': 'Times New Roman', 'size': font_size, 'weight': weight})
    plt.xlabel('Design variable',
               fontdict={'family': 'Times New Roman', 'size': font_size, 'weight': weight})
    plt.legend(['Before opt', 'After 5 sims', 'After opt'], loc=3, prop={'family': 'Times New Roman', 'size': 18, 'weight': 'normal'})
    plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.21)
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
            cur_wavelength, p11, p12, p13, p14 = [float(ele) for ele in line.strip().split("\t")]
            res.append([cur_wavelength, p11, p12, p13, p14])
    res = np.array(res)
    plt.figure()
    plt.plot(res[:, 0], res[:, 2], color='blue', linewidth=ld, label=r'$p_{12}$')
    plt.plot(res[:, 0], res[:, 4], color='orange', linewidth=ld, label=r'$p_{14}$')
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
    plt.yticks([0, 0.5, 1.0], font='Times New Roman', size=font_size)
    plt.xticks([1.50, 1.55, 1.60], font='Times New Roman', size=font_size)
    plt.legend(prop={'family': 'Times New Roman', 'size': legend_size, 'weight': weight})

    plt.show()


if __name__ == '__main__':
    # care_iter = [60]
    myplot([4,59])
    # myplot2()
    # w0 = 0.6e-6 * np.ones(13, )
    # data_tranform(w0)
    # cur_best_w = np.load('./result/cur_best_w_LCB1_0.npy')
    # print(cur_best_w)
    # data_tranform(cur_best_w)  # generate a design_vari.mat in the current folder based on the obtained result.
