from matplotlib import pyplot as plt
import numpy as np

select_length = 70

plt.rcParams["font.family"] = "Times New Roman"

Y_LCB = np.load("./result/Y_LCB1_0.npy")
X_LCB = np.load("./result/X_LCB1_0.npy")
# cur_best_w_LCB = np.load('./result/cur_best_w_LCB1_0.npy')

Y_LCB, X_LCB = Y_LCB[:select_length], X_LCB[:select_length, :]

Y_LCB2 = np.load("./result/Y_lCB2_0.npy")
X_LCB2 = np.load("./result/X_LCB2_0.npy")
cur_best_w_LCB2 = np.load('./result/cur_best_w_LCB2_0.npy')

Y_LCB2, X_LCB2 = Y_LCB2[:select_length], X_LCB2[:select_length, :]

Y_LCB3 = np.load("./result/Y_lCB3_0.npy")
X_LCB3 = np.load("./result/X_LCB3_0.npy")
cur_best_w_LCB3 = np.load('./result/cur_best_w_LCB3_0.npy')

Y_LCB3, X_LCB3 = Y_LCB3[:select_length], X_LCB3[:select_length, :]

Y_ei = np.load("./result/Y_EI_0.npy")
X_ei = np.load("./result/X_EI_0.npy")
# cur_best_w_ei = np.load('./result/cur_best_w_EI_0.npy')

Y_ei, X_ei = Y_ei[:select_length], X_ei[:select_length, :]


Y_pi = np.load("./result/Y_PI_0.npy")
X_pi = np.load("./result/X_PI_0.npy")
# cur_best_w_pi = np.load('./result/cur_best_w_PI_0.npy')

Y_pi, X_pi = Y_pi[:select_length], X_pi[:select_length, :]

tmp_lcb = [np.min(Y_LCB[0:i]) for i in range(1, Y_LCB.shape[0]+1)]
tmp_lcb2 = [np.min(Y_LCB2[0:i]) for i in range(1, Y_LCB2.shape[0])]
tmp_lcb3 = [np.min(Y_LCB3[0:i]) for i in range(1, Y_LCB3.shape[0])]

tmp_ei = [np.min(Y_ei[0:i]) for i in range(1, Y_ei.shape[0]+1)]
tmp_pi = [np.min(Y_pi[0:i]) for i in range(1, Y_pi.shape[0]+1)]

print(tmp_lcb[select_length-1], tmp_ei[select_length-1], tmp_pi[select_length-1])

# print(len(tmp_lcb))
# print(tmp_lcb[29])
# print(tmp_lcb[49])
# print(tmp_lcb[69])
# print(tmp_lcb[99])
# print(tmp_lcb[129])
# print(tmp_lcb[-1])
#
# print(tmp_ei[69])
# print(tmp_pi[69])

markersize = 10
font_size = 32
legend_size = 28
weight = 'normal'
ld = 3

fig = plt.figure(figsize=[8, 6])
plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
plt.plot(list(range(len(tmp_lcb))), tmp_lcb, color='orange', linewidth=3, label='LCB')
plt.plot(list(range(len(tmp_ei))), tmp_ei, color='blue', linewidth=3, label='EI')
plt.plot(list(range(len(tmp_pi))), tmp_pi, color='green', linewidth=3, label='PI')
plt.yticks([-0.0, 0.03,0.06], font='Times New Roman', size=font_size)
plt.xticks([0, 40, 80], font='Times New Roman', size=font_size)
plt.ylabel('Objective value', fontdict={'family': 'Times New Roman', 'size': font_size, 'weight': weight})
plt.xlabel('# simulation',
           fontdict={'family': 'Times New Roman', 'size': font_size, 'weight': weight})
plt.legend(prop={'family': 'Times New Roman', 'size': 24, 'weight': 'normal'})

ax = plt.gca()
ax.spines['bottom'].set_linewidth(ld)
ax.spines['left'].set_linewidth(ld)
ax.spines['right'].set_linewidth(ld)
ax.spines['top'].set_linewidth(ld)

fig = plt.figure(figsize=[8, 6])
# plt.plot(list(range(len(Y_LCB))), Y_LCB, color='orange', linewidth=3, label='LCB')
plt.plot(list(range(len(Y_ei))), Y_ei, color='blue', linewidth=3, label='EI')
# plt.plot(list(range(len(Y_pi))), Y_pi, color='green', linewidth=3, label='PI')

plt.yticks([-0.0, 0.08, 0.16], font='Times New Roman', size=font_size)
plt.xticks([0, 40, 80], font='Times New Roman', size=font_size)
plt.ylabel('Objective value', fontdict={'family': 'Times New Roman', 'size': font_size, 'weight': weight})
plt.xlabel('# simulation',
           fontdict={'family': 'Times New Roman', 'size': font_size, 'weight': weight})
plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
plt.legend(prop={'family': 'Times New Roman', 'size': 24, 'weight': 'normal'})

ax = plt.gca()
ax.spines['bottom'].set_linewidth(ld)
ax.spines['left'].set_linewidth(ld)
ax.spines['right'].set_linewidth(ld)
ax.spines['top'].set_linewidth(ld)


fig = plt.figure(figsize=[8, 6])
plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
plt.plot(list(range(len(tmp_lcb))), tmp_lcb, color='orange', linewidth=3, label=r'LCB $\gamma$=0.3')
plt.plot(list(range(len(tmp_lcb2))), tmp_lcb2, color='blue', linewidth=3, label=r'LCB $\gamma$=1.0')
plt.plot(list(range(len(tmp_lcb3))), tmp_lcb3, color='green', linewidth=3, label=r'LCB $\gamma$=2.0')
plt.yticks([0, 0.03, 0.06], font='Times New Roman', size=font_size)
plt.xticks([0, 40, 80], font='Times New Roman', size=font_size)
plt.ylabel('Objective value', fontdict={'family': 'Times New Roman', 'size': font_size, 'weight': weight})
plt.xlabel('# simulation',
           fontdict={'family': 'Times New Roman', 'size': font_size, 'weight': weight})
plt.legend(prop={'family': 'Times New Roman', 'size': 24, 'weight': 'normal'})

ax = plt.gca()
ax.spines['bottom'].set_linewidth(ld)
ax.spines['left'].set_linewidth(ld)
ax.spines['right'].set_linewidth(ld)
ax.spines['top'].set_linewidth(ld)
plt.show()
