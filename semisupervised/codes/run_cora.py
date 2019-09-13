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

data = np.loadtxt('../data/cora/label.txt').astype(int)
labels = data[:,1]
n_classes = len(np.unique(labels))

(indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_classes)]))
# Ensure uniform distribution of labels
np.random.shuffle(indices)
        
n = 10
n_valid = 10
n_test = 10
indices_train = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(n_classes)])
indices_valid = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[n:n+n_valid] for i in range(n_classes)])
indices_test = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[n+n_valid:n+n_valid+n_test] for i in range(n_classes)])

np.random.shuffle(indices_train)
np.random.shuffle(indices_valid)
np.random.shuffle(indices_test)



if os.path.exists('../data/cora/train_temp.txt'):
        os.remove('../data/cora/train_temp.txt')
        os.remove('../data/cora/dev_temp.txt')
        os.remove('../data/cora/test_temp.txt')

np.savetxt('../data/cora/train_temp.txt', indices_train, fmt='%d')
np.savetxt('../data/cora/dev_temp.txt', indices_valid, fmt='%d')
np.savetxt('../data/cora/test_temp.txt', indices_test, fmt='%d')

### data creation ended ####


print('mixup_alpha_'+opt['mixup_alpha'])
print('mixup_consistency_'+opt['mixup_consistency'])

run(opt)

