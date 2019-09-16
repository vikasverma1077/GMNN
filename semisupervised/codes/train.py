import sys
import os
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, auc

from trainer import Trainer
from gcn import GCN
import loader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='data')
parser.add_argument('--save', type=str, default='/')
parser.add_argument('--hidden_dim', type=int, default=16, help='RNN hidden state size.')
parser.add_argument('--input_dropout', type=float, default=0.5, help='Input and RNN dropout rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Input and RNN dropout rate.')
parser.add_argument('--optimizer', type=str, default='adam', help='Optional info for the experiment.')
parser.add_argument('--lr', type=float, default=0.01, help='Applies to SGD and Adagrad.')
parser.add_argument('--decay', type=float, default=5e-4)
parser.add_argument('--self_link_weight', type=float, default=1.0)
parser.add_argument('--pre_epoch', type=int, default=200)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--iter', type=int, default=10)
parser.add_argument('--use_gold', type=int, default=1)
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--draw', type=str, default='max')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

opt = vars(args)

# load data
print("Loading data from {}...".format(opt['dataset']))
net_file = opt['dataset'] + '/net.txt'
label_file = opt['dataset'] + '/label.txt'
feature_file = opt['dataset'] + '/feature.txt'
train_file = opt['dataset'] + '/train.txt'
dev_file = opt['dataset'] + '/dev.txt'
test_file = opt['dataset'] + '/test.txt'

vocab_node = loader.Vocab(net_file, [0, 1])
vocab_label = loader.Vocab(label_file, [1])
vocab_feature = loader.Vocab(feature_file, [1])

opt['num_node'] = len(vocab_node)
opt['num_feature'] = len(vocab_feature)
opt['num_class'] = len(vocab_label)

graph = loader.Graphs(file_name=net_file, entity=[vocab_node, 0, 1])
label = loader.EntityLabel(file_name=label_file, entity=[vocab_node, 0], label=[vocab_label, 1])
feature = loader.EntityFeature(file_name=feature_file, entity=[vocab_node, 0], feature=[vocab_feature, 1])
graph.to_symmetric(opt['self_link_weight'])
feature.to_one_hot()

slices = graph.get_slices()

idx_train, idx_dev, idx_test = [], [], []

fi = open(train_file, 'r')
idx_train = [vocab_node.stoi[line.strip()] for line in fi]
fi.close()

fi = open(dev_file, 'r')
idx_dev = [vocab_node.stoi[line.strip()] for line in fi]
fi.close()

fi = open(test_file, 'r')
idx_test = [vocab_node.stoi[line.strip()] for line in fi]
fi.close()

idx_all = list(range(opt['num_node']))

# model
# initialize a relation model
if opt['cuda']:
    slices = [s.cuda() for s in slices]

gcn = GCN(opt, slices)
trainer_gcn = Trainer(opt, gcn)

inputs = torch.Tensor(feature.one_hot)
target = torch.LongTensor(label.itol)
idx_train = torch.LongTensor(idx_train)
idx_dev = torch.LongTensor(idx_dev)
idx_test = torch.LongTensor(idx_test)
idx_all = torch.LongTensor(idx_all)

if opt['cuda']:
    inputs = inputs.cuda()
    target = target.cuda()
    idx_train = idx_train.cuda()
    idx_dev = idx_dev.cuda()
    idx_test = idx_test.cuda()
    idx_all = idx_all.cuda()

inputs_gcn = torch.zeros(opt['num_node'], opt['num_feature'])
target_gcn = torch.zeros(opt['num_node'], opt['num_class'])

if opt['cuda']:
    inputs_gcn = inputs_gcn.cuda()
    target_gcn = target_gcn.cuda()

def init_gcn_data():
    inputs_gcn.copy_(inputs)
    temp = torch.zeros(idx_train.size(0), target_gcn.size(1)).type_as(target_gcn)
    temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
    target_gcn[idx_train] = temp

def auc_data(x, y):
	z = zip(x, y)
	z = sorted(z, key=lambda a:a[0], reverse=True)
	x, y = zip(*z)
	return np.array(x), np.array(y)

def pre_train(epoches):

    init_gcn_data()

    results = []

    for epoch in range(epoches):

        loss = trainer_gcn.update_soft(inputs_gcn, target_gcn, idx_train)

        _, preds, accuracy_dev = trainer_gcn.evaluate(inputs_gcn, target, idx_dev)
        f1_dev = f1_score(target[idx_dev].cpu().numpy(), preds.cpu().numpy(), average='macro')

        _, preds, accuracy_test = trainer_gcn.evaluate(inputs_gcn, target, idx_test)
        f1_test = f1_score(target[idx_test].cpu().numpy(), preds.cpu().numpy(), average='macro')

        results += [(-loss, f1_test)]

    return results

base_results = []
base_results += pre_train(opt['pre_epoch'])

def final_numbers(results):
    bk, bd, bt = 0, 0.0, 0.0
    for k, (d, t) in enumerate(results):
        if d > bd:
            bk, bd, bt = k, d, t
    k = min(bk, len(results)-1)
    return results[k][0], results[k][1]

d, t = final_numbers(base_results)

print('Final F1: {:.3f}'.format(t * 100))

