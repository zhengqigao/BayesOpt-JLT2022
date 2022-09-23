from matplotlib import pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

# Y_LCB = np.load("./result/Y_LCB1_0.5_0_cubic.npy")
# X_LCB = np.load("./result/X_LCB1_0.5_0_cubic.npy")
# acq_list = np.load("./result/acq_list_LCB1_0.5_0_cubic.npy")
# cur_best_w_LCB = np.load('./result/cur_best_w_LCB1_0.5_0_cubic.npy')

Y_LCB = np.load("./result/Y_LCB1_0.npy")
X_LCB = np.load("./result/X_LCB1_0.npy")
acq_list = np.load("./result/acq_list_LCB1_0.npy")[1:]
cur_best_w_LCB = np.load('./result/cur_best_w_LCB1_0.npy')


tmp_lcb = [np.min(Y_LCB[0:i]) for i in range(1, Y_LCB.shape[0]+1)]
tmp_acq = [np.min(acq_list[0:i]) for i in range(1, acq_list.shape[0]+1)]

print(len(tmp_lcb))
print(tmp_lcb[-1])

markersize = 10
font_size = 32
legend_size = 28
weight = 'normal'
ld = 3

fig = plt.figure(figsize=[8, 6])
plt.subplots_adjust(left=0.22, right=0.9, top=0.9, bottom=0.15)
plt.plot(list(range(len(tmp_lcb))), tmp_lcb, color='orange', linewidth=3, label='LCB')
# plt.yticks([0, 0.02, 0.04], font='Times New Roman', size=font_size)
# plt.xticks([0, 40, 80, 120], font='Times New Roman', size=font_size)
plt.ylabel('Objective value', fontdict={'family': 'Times New Roman', 'size': font_size, 'weight': weight})
plt.xlabel('# simulation',
           fontdict={'family': 'Times New Roman', 'size': font_size, 'weight': weight})
plt.legend(prop={'family': 'Times New Roman', 'size': 24, 'weight': 'normal'})
ax = plt.gca()
ax.spines['bottom'].set_linewidth(ld)
ax.spines['left'].set_linewidth(ld)
ax.spines['right'].set_linewidth(ld)
ax.spines['top'].set_linewidth(ld)

print(acq_list)

fig = plt.figure(figsize=[8, 6])
plt.subplots_adjust(left=0.24, right=0.9, top=0.9, bottom=0.20)
plt.plot(list(range(len(tmp_lcb)-len(tmp_acq), len(tmp_lcb))), -np.array(tmp_acq), color='orange', linewidth=3, label='LCB')
plt.yticks([-0.008, -0.003, 0.002], font='Times New Roman', size=font_size)
plt.xticks([30, 50, 70], font='Times New Roman', size=font_size)
plt.ylabel('Optimal LCB Value', fontdict={'family': 'Times New Roman', 'size': font_size, 'weight': weight})
plt.xlabel('# simulation',
           fontdict={'family': 'Times New Roman', 'size': font_size, 'weight': weight})
plt.legend(prop={'family': 'Times New Roman', 'size': 24, 'weight': 'normal'})
ax = plt.gca()
ax.spines['bottom'].set_linewidth(ld)
ax.spines['left'].set_linewidth(ld)
ax.spines['right'].set_linewidth(ld)
ax.spines['top'].set_linewidth(ld)

fig = plt.figure(figsize=[8, 6])
plt.subplots_adjust(left=0.22, right=0.9, top=0.9, bottom=0.20)
plt.plot(list(range(len(tmp_lcb)-len(tmp_acq), len(tmp_lcb))), -np.array(acq_list), color='orange', linewidth=3, label='LCB')
plt.yticks([-0.02, -0.01, 0.0], font='Times New Roman', size=font_size)
plt.xticks([30, 50, 70], font='Times New Roman', size=font_size)
plt.ylabel('Current LCB Value', fontdict={'family': 'Times New Roman', 'size': font_size, 'weight': weight})
plt.xlabel('# simulation',
           fontdict={'family': 'Times New Roman', 'size': font_size, 'weight': weight})
plt.legend(prop={'family': 'Times New Roman', 'size': 24, 'weight': 'normal'})
ax = plt.gca()
ax.spines['bottom'].set_linewidth(ld)
ax.spines['left'].set_linewidth(ld)
ax.spines['right'].set_linewidth(ld)
ax.spines['top'].set_linewidth(ld)


plt.show()
