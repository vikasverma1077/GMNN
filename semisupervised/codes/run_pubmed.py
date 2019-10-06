import sys
import os
import copy
import json
import datetime

opt = dict()

opt['dataset'] = '../data/pubmed'
opt['hidden_dim'] = 8
opt['input_dropout'] = 0.2
opt['dropout'] = 0.2
opt['optimizer'] = 'adam'
opt['lr'] = 0.005
opt['decay'] = 5e-4
opt['pre_epoch'] = 2000
opt['epoch'] = 100
opt['iter'] = 1
opt['use_gold'] = 1
opt['draw'] = 'smp'
opt['tau'] = 0.1
opt['save'] = 'exp_pubmed'
opt['mixup_alpha'] = 0.1

### ict hyperparameters ###
opt['ema_decay'] = 0.999
opt['consistency_type'] = "mse"
opt['consistency_rampup_starts'] = 500
opt['consistency_rampup_ends'] = 1000
opt['mixup_consistency'] = 10


def generate_command(opt):
    cmd = 'python3 train.py'
    for opt, val in opt.items():
        cmd += ' --' + opt + ' ' + str(val)
    return cmd

def run(opt):
    opt_ = copy.deepcopy(opt)
    os.system(generate_command(opt_))

for k in range(20):
    seed = k + 1
    opt['seed'] = seed
    run(opt)
