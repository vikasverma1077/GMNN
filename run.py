import sys
import os
import copy
import json
import datetime

opt = dict()

opt['dataset'] = 'ppi'
opt['hidden_dim'] = 512
opt['num_sample'] = 25
opt['batch_size'] = 512
#opt['input_dropout'] = 0
opt['dropout'] = 0
opt['optimizer'] = 'adam'
opt['lr'] = 0.005
opt['decay'] = 0# 5e-4
opt['tau'] = 1
opt['self_link_weight'] = 0.0
opt['epoch'] = 1000
opt['gcn'] = 0
opt['mixup_alpha'] = 1 #1 80 2 79.6 0.5 78.7 0.1 78.3 100 79.8

opt['ema_decay'] = 0.999
opt['consistency_type'] = "mse"
opt['consistency_rampup_starts'] = 500
opt['consistency_rampup_ends'] = 1000
opt['mixup_consistency'] = 1

def generate_command(opt):
    cmd = 'python3 train.py'
    for opt, val in opt.items():
        cmd += ' --' + opt + ' ' + str(val)
    return cmd

def run(opt):
    opt_ = copy.deepcopy(opt)
    os.system(generate_command(opt_))

for k in range(1):
    seed = k + 1
    opt['seed'] = seed
    run(opt)
