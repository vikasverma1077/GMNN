import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import argparse
import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from graphsage import loader
from graphsage.layer import mean_aggregator, graphsage_encoder
from graphsage.gcn import GSGCN
from graphsage.trainer import Trainer

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='data')
parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension.')
parser.add_argument('--num_sample', type=int, default=5, help='Number of samples.')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
parser.add_argument('--self_link_weight', type=float, default=1.0, help='Weight of self-links.')
parser.add_argument('--mixup_alpha', type=float, default=1, help='alpha for mixing')
parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs per iteration.')
parser.add_argument('--gcn', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

opt = vars(args)

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

graph = loader.Graph(file_name=net_file, entity=[vocab_node, 0, 1])
label = loader.EntityFeature(file_name=label_file, entity=[vocab_node, 0], feature=[vocab_label, 1])
feature = loader.EntityFeature(file_name=feature_file, entity=[vocab_node, 0], feature=[vocab_feature, 1])

graph.build_neighbor()
label.to_one_hot(normalize=False)
feature.to_one_hot()
adj_lists = graph.nbs

with open(train_file, 'r') as fi:
    idx_train = [vocab_node.stoi[line.strip()] for line in fi]
with open(dev_file, 'r') as fi:
    idx_dev = [vocab_node.stoi[line.strip()] for line in fi]
with open(test_file, 'r') as fi:
    idx_test = [vocab_node.stoi[line.strip()] for line in fi]

inputs = torch.Tensor(feature.one_hot)
target = torch.Tensor(label.one_hot)
idx_train = torch.LongTensor(idx_train)
idx_dev = torch.LongTensor(idx_dev)
idx_test = torch.LongTensor(idx_test)

idx_train = idx_train[torch.randperm(idx_train.size(0))]
idx_train_double = torch.cat([idx_train, idx_train], dim=0)

graphsage = GSGCN(opt, inputs, adj_lists)
trainer = Trainer(opt, graphsage)

def compute_f1(true, pred):
    true = true.data.numpy()
    pred = pred.data.cpu().numpy()

    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0

    return f1_score(true, pred, average="micro")

results = []
for epoch in range(opt['epoch']):
    bg = int(random.random() * idx_train.size(0))
    ed = bg + opt['batch_size']

    batch_nodes = idx_train_double[bg:ed]
    #idx_train = idx_train[torch.randperm(idx_train.size(0))]
    #start_time = time.time()
    #loss = trainer.update_gs(batch_nodes, target)
    #end_time = time.time()
    #times.append(end_time-start_time)

    rand_index = random.randint(0,1)
    rand_index = 1# if random.random() < 0.9 else 0
    if rand_index == 0: ## do the augmented node training
        loss = trainer.update_soft_aux(batch_nodes, target, mixup_layer =[1])## for augmented nodes
        trainer.model.train()
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
    else:
        loss = trainer.update_gs(batch_nodes, target)

    dev_output = torch.sigmoid(graphsage.forward(idx_dev))
    dev_f1 = compute_f1(target[idx_dev], dev_output)
    test_output = torch.sigmoid(graphsage.forward(idx_test))
    test_f1 = compute_f1(target[idx_test], test_output)

    results += [(dev_f1, test_f1)]

    print('Epoch: {} | Loss: {:.5f} | Dev F1: {:.3f} | Test F1: {:.3f}'.format(epoch, loss, dev_f1, test_f1))

bd, bt = 0, 0
for d, t in results:
	if d > bd:
		bd = d
		bt = t
print('Test F1: {:.3f}'.format(bt))
