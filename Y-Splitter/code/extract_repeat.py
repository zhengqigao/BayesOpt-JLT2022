import numpy as np

file_prefix1 = './logs/log_lcb1_seed'
file_prefix2 = './logs/log_ga_seed'

list1 = list(range(0,20))
list2 = list(range(0,5))

res1 = []
res2 = []

for i in list1:
    file_name = file_prefix1 + str(i) + '.txt'
    with open(file_name, 'r') as f:
        for line in f.readlines():
            wrk = line
    wrk2 = wrk.split(' optimal metrics:')
    res1.append(eval(wrk2[-1].strip()))

for i in list2:
    file_name = file_prefix2 + str(i) + '.txt'
    with open(file_name, 'r') as f:
        wrk = f.readlines()
    wrk2 = wrk[0].split('=')
    res2.append(eval(wrk2[-1]))


res1 = np.array(res1)
res2 = np.array(res2)

print(res1.mean())
print(res2.mean())


print(res1.max())
print(res2.max())

print(res1.min())
print(res2.min())

print(res1.std())
print(res2.std())

print(res1<=7.4E-4)