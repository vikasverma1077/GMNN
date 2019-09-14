import numpy as np
from operator import __or__

data = np.loadtxt('../data/cora/label.txt').astype(int)
labels = data[:,1]
n_classes = len(np.unique(labels))


n = 10
n_valid = 10
n_test = 100


for i in np.arange(n_classes):
    idx= data[labels==i,0]
    np.random.shuffle(idx)
    
    if i==0:
        indices_train = idx[:n]
        indices_valid = idx[n:n+n_valid]
        indices_test = idx[n+n_valid:n+n_valid+n_test]
    else:
        indices_train = np.hstack((indices_train,idx[:n]))
        indices_valid = np.hstack((indices_valid,idx[n:n+n_valid]))
        indices_test = np.hstack((indices_test,idx[n+n_valid:n+n_valid+n_test]))
    

np.random.shuffle(indices_train)
np.random.shuffle(indices_valid)
np.random.shuffle(indices_test)

np.savetxt('../data/cora/train_temp.txt', indices_train, fmt='%d')
np.savetxt('../data/cora/dev_temp.txt', indices_valid, fmt='%d')
np.savetxt('../data/cora/test_temp.txt', indices_test, fmt='%d')