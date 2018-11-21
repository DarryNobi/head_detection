import matplotlib.pyplot as plt
import numpy as np

def movingaverage(data, window_size):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return ma_vec


f = open('acc_regular.txt')
contents = f.read()
f.close()
acc=contents.split(' ')
acc=[float(i) for i in acc]
acc=movingaverage(acc,20)

f = open('acc_fuzzy.txt')
contents_fuzzy = f.read()
f.close()
acc_fuzzy=contents_fuzzy.split(' ')
acc_fuzzy=[float(i) for i in acc_fuzzy]
acc_fuzzy=movingaverage(acc_fuzzy,20)

plt.ylim(0,1)
plt.plot(acc,'-r')
plt.plot(acc_fuzzy,'-g')
plt.show()

