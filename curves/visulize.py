import matplotlib.pyplot as plt
import numpy as np

def movingaverage(data, window_size):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return ma_vec

def  plot_file(file,sli_win=40,sample=10):
    f = open(file)
    contents = f.read()
    f.close()
    acc=contents.split(' ')
    acc=[float(i) for i in acc]
    acc=movingaverage(acc[0:500],sli_win)
    return acc[0:-1:sample]


plt.figure(figsize=(10, 5))
plt.grid(linestyle="--")  # 设置背景网格线为虚线
ax = plt.gca()
ax.spines['top'].set_visible(False)  # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框

plt.xlabel("Iterations", fontsize=13, fontweight='bold')
plt.ylabel("Accuracy", fontsize=13, fontweight='bold')


plt.ylim(0,1)
# plt.plot(plot_file('test_acc_fuzzy1228.txt'),'-r',marker='*',label='fuzzy')
# plt.plot(plot_file('test_acc_fuzzy1229.txt'),'-g',marker='.',label='offset')
plt.plot(plot_file('test_acc_googlenet.txt'),'-y',marker='.',label='googlenet')
plt.plot(plot_file('test_acc_resnet1229.txt'),'-b',label='resnet')
# plt.plot(plot_file('acc_fuzzy_onlyoffsetloss.txt'),'-y')
# plt.plot(plot_file('test_acc_fuzzyonlyoffsetloss.txt'),'-r')
# plt.plot(plot_file('test_acc_tmp.txt'),'-y')
# plt.plot(plot_file('test_acc_fuzzy_fc.txt'),'-g')
# plt.plot(plot_file('tmp.txt'),'-g')
# plt.plot(plot_file('test_acc_naive.txt'),'-g')



group_labels = ['1k', '2k', '3k', '4k', '5k']
plt.xticks([10,20,30,40,50], group_labels, fontsize=12, fontweight='bold')  # 默认字体大小为10
plt.yticks(fontsize=12, fontweight='bold')
plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=12, fontweight='bold')



plt.show()

