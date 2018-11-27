import matplotlib.pyplot as plt
import numpy as np

def movingaverage(data, window_size):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return ma_vec


f = open('test_acc_fuzzy.txt')
contents = f.read()
f.close()
acc=contents.split(' ')
acc=[float(i) for i in acc]
acc=movingaverage(acc,30)
# acc=acc[0:-1:3]

f = open('acc_fuzzy.txt')
contents_fuzzy = f.read()
f.close()
acc_fuzzy=contents_fuzzy.split(' ')
acc_fuzzy=[float(i) for i in acc_fuzzy]
acc_fuzzy=movingaverage(acc_fuzzy,30)
# acc_fuzzy=acc_fuzzy[0:-1:3]

f = open('acc_offset.txt')
contents_offset = f.read()
f.close()
acc_offset=contents_offset.split(' ')
acc_offset=[float(i) for i in acc_offset]
acc_offset=movingaverage(acc_offset,30)

plt.ylim(0,1)
plt.plot(acc,'-r')
plt.plot(acc_fuzzy,'-g')
# plt.plot(acc_offset,'-y')
plt.show()

