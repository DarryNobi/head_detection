import matplotlib.pyplot as plt
import numpy as np

def movingaverage(data, window_size):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return ma_vec

def  plot_file(file,sli_win=60):
    f = open(file)
    contents = f.read()
    f.close()
    acc=contents.split(' ')
    acc=[float(i) for i in acc]
    acc=movingaverage(acc,sli_win)
    return acc




plt.ylim(0,1)
plt.plot(plot_file('test_acc_fuzzy1221.txt'),'-r')
plt.plot(plot_file('acc_fuzzy_onlyoffsetloss.txt'),'-y')
# plt.plot(plot_file('test_acc_fuzzyonlyoffsetloss.txt'),'-r')
# plt.plot(plot_file('test_acc_tmp.txt'),'-y')
# plt.plot(plot_file('test_acc_fuzzy_fc.txt'),'-g')
# plt.plot(plot_file('test_acc_fuzzy2.txt'),'-g')
# plt.plot(plot_file('test_acc_fuzzy.txt'),'-g')
plt.show()

