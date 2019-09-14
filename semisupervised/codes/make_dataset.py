import numpy as np
from operator import __or__
import os
from distutils.dir_util import copy_tree


data = np.loadtxt('../data/pubmed/label.txt').astype(int)
labels = data[:,1]
n_classes = len(np.unique(labels))



n = 20
n_valid = 167
#n_test = 100

for j in range(5):
    new_data_dir = os.path.join('../data_subset/pubmed/n20', str(j+1))
    os.makedirs(new_data_dir)
    copy_tree('../data/pubmed', new_data_dir)

    np.random.seed(j)

    for i in np.arange(n_classes):
        idx= data[labels==i,0]
        np.random.shuffle(idx)
    
        if i==0:
            indices_train = idx[:n]
            indices_valid = idx[n:n+n_valid]
            indices_test = idx[n+n_valid:]
        else:
            indices_train = np.hstack((indices_train,idx[:n]))
            indices_valid = np.hstack((indices_valid,idx[n:n+n_valid]))
            indices_test = np.hstack((indices_test,idx[n+n_valid:]))
    

    np.random.shuffle(indices_train)
    np.random.shuffle(indices_valid)
    np.random.shuffle(indices_test)

    np.savetxt(os.path.join(new_data_dir,'train_temp.txt'), indices_train, fmt='%d')
    np.savetxt(os.path.join(new_data_dir,'dev_temp.txt'), indices_valid, fmt='%d')
    np.savetxt(os.path.join(new_data_dir,'test_temp.txt'), indices_test, fmt='%d')
