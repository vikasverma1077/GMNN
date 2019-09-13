import sys
import os
import copy
import json
import datetime

opt = dict()

opt['dataset'] = '../data/cora'
opt['hidden_dim'] = 16
opt['input_dropout'] = 0.5
opt['dropout'] = 0
opt['optimizer'] = 'adam'
opt['lr'] = 0.01
opt['decay'] = 5e-4
opt['self_link_weight'] = 1.0
opt['pre_epoch'] = 2000
opt['epoch'] = 100
opt['iter'] = 1
opt['use_gold'] = 1
opt['draw'] = 'smp'
opt['tau'] = 0.0
opt['save'] = 'exp_cora'
opt['mixup_alpha'] =1.0


### ict hyperparameters ###
opt['ema_decay'] = 0.999
opt['consistency_type'] = "mse"
opt['consistency_rampup_starts'] = 500
opt['consistency_rampup_ends'] = 1000
opt['mixup_consistency'] = 10.0



def generate_command(opt):
    cmd = 'python3 train.py'
    for opt, val in opt.items():
        cmd += ' --' + opt + ' ' + str(val)
    return cmd

def run(opt):
    opt_ = copy.deepcopy(opt)
    os.system(generate_command(opt_))

"""
for k in range(5):
    seed = k + 1
    opt['seed'] = seed

    print('mixup_alpha_'+opt['mixup_alpha'])
    print('mixup_consistency_'+opt['mixup_consistency'])

    run(opt)
    
"""

### create subset data###
import numpy as np
from operator import __or__
from functools import reduce

data = np.loadtxt('../data/cora/label.txt').astype(int)
labels = data[:,1]
n_classes = len(np.unique(labels))


n = 15
n_valid = 15
n_test = 100
print (n, n_valid, n_test)

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

### data creation ended ####

for i in [0.1, 1.0, 2.0]:
    for j in [1.0, 10.0, 20.0]:
        opt['mixup_alpha'] = i
        opt['mixup_consistency'] = j
        print('mixup_alpha_'+str(opt['mixup_alpha']))
        print('mixup_consistency_'+ str(opt['mixup_consistency']))

        run(opt)

