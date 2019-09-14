import sys
import os
import copy
import json
import datetime

opt = dict()

opt['dataset'] = '../data/citeseer'
opt['hidden_dim'] = 8
opt['input_dropout'] = 0.5
opt['dropout'] = 0.6
opt['optimizer'] = 'adam'
opt['lr'] = 0.005
opt['decay'] = 5e-4
opt['self_link_weight'] = 1.0
opt['pre_epoch'] = 2000
opt['tau'] = 0.1
opt['save'] = 'exp_cora'

### ict hyperparameters ###

opt['mixup_alpha'] = 1.0

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


for k in range(0, 10):
    seed = k + 1

    run(opt)

                            #exit()
