import sys
import os
import copy
import json
import datetime

file = 'train.py'
path = '../saved_model'

opt = dict()

opt['dataset'] = '../data/bitotc'
opt['hidden_dim'] = 128
opt['input_dropout'] = 0.0
opt['dropout'] = 0
opt['optimizer'] = 'adam'
opt['lr'] = 0.01
opt['decay'] = 0#5e-4
opt['self_link_weight'] = 1.0
opt['pre_epoch'] = 15
opt['epoch'] = 5
opt['iter'] = 5
opt['use_gold'] = 1
opt['draw'] = 'smp'
opt['gamma'] = 10

opt['log_step'] = 1000
opt['seed'] = 1


opt['mixup_alpha'] = 0.1


### ict hyperparameters ###
opt['ema_decay'] = 0.999
opt['consistency_type'] = "mse"
opt['consistency_rampup_starts'] = 5
opt['consistency_rampup_ends'] = 10
opt['mixup_consistency'] = 0.1

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def generate_command(opt):
    cmd = 'python3 ' + file
    for opt, val in opt.items():
        cmd += ' --' + opt + ' ' + str(val)
    return cmd

def save_option(opt, path):
    with open(path, 'w') as fo:
        for key, val in opt.items():
            fo.write(key + '\t' + str(val) + '\n')

def run(opt):
    opt_ = copy.deepcopy(opt)
    #time = str(datetime.datetime.now()).replace('-', '').replace(' ', '').replace(':', '').replace('.', '')
    #local_path = path + '/' + time
    #ensure_dir(local_path)
    #save_option(opt_, local_path + '/opt.txt')
    #opt_['save'] = local_path
    os.system(generate_command(opt_))

for k in range(1):
    seed = k + 1
    opt['seed'] = seed

    print(opt['mixup_alpha'])
    print(opt['mixup_consistency'])

    run(opt)


