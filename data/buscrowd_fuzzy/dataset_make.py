import os
import csv
import numpy as np

def make_train_set(path):
    data_dict=[]
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            lable=[dirname[0],dirname[1]]
            for path,dir,files in os .walk(os.path.join(dirpath,dirname)):
                for file in files:
                    data_dict.append([os.path.join(path,file),lable[0],lable[1]])
    csvFile = open('train.csv', 'w', newline='')  # 设置newline，否则两行之间会空一行
    writer = csv.writer(csvFile)
    m = len(data_dict)
    for i in range(m):
        writer.writerow(data_dict[i])
    csvFile.close()
def main():
    make_train_set('train')

if __name__=='__main__':
    main()