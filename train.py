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

from ramps import *

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
parser.add_argument('--tau', type=float, default=0.1, help='Dropout rate.')
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

parser.add_argument('--ema_decay', default=0.999, type=float, metavar='ALPHA',
                    help='ema variable decay rate (default: 0.999)')
#parser.add_argument('--consistency', default=None, type=float, metavar='WEIGHT',
#                    help='use consistency loss with given weight (default: None)')
parser.add_argument('--consistency_type', default="mse", type=str, metavar='TYPE',
                    choices=['mse', 'kl'],
                    help='consistency loss type to use')
parser.add_argument('--consistency_rampup_starts', default=30, type=int, metavar='EPOCHS',
                    help='epoch at which consistency loss ramp-up starts')
parser.add_argument('--consistency_rampup_ends', default=30, type=int, metavar='EPOCHS',
                    help='lepoch at which consistency loss ramp-up ends')
#parser.add_argument('--mixup_sup_alpha', default=0.0, type=float,
#                    help='for supervised loss, the alpha parameter for the beta distribution from where the mixing lambda is drawn')
#parser.add_argument('--mixup_usup_alpha', default=0.0, type=float,
#                    help='for unsupervised loss, the alpha parameter for the beta distribution from where the mixing lambda is drawn')
#parser.add_argument('--mixup_hidden', action='store_true',
#                    help='apply mixup in hidden layers')
#parser.add_argument('--num_mix_layer', default=3, type=int,
#                    help='number of hidden layers on which mixup is applied in addition to input layer')
parser.add_argument('--mixup_consistency', default=1.0, type=float,
                    help='max consistency coeff for mixup usup loss')

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
unlabeled_file = opt['dataset'] + '/unlabeled.txt'

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
with open(unlabeled_file, 'r') as fi:
    idx_unlabeled = [vocab_node.stoi[line.strip()] for line in fi]

inputs = torch.Tensor(feature.one_hot)
target = torch.Tensor(label.one_hot)
idx_train = torch.LongTensor(idx_train)
idx_dev = torch.LongTensor(idx_dev)
idx_test = torch.LongTensor(idx_test)
idx_unlabeled = torch.LongTensor(idx_unlabeled)

idx_train = idx_train[torch.randperm(idx_train.size(0))]
idx_train_double = torch.cat([idx_train, idx_train], dim=0)

idx_unlabeled = idx_unlabeled[torch.randperm(idx_unlabeled.size(0))]
idx_unlabeled_double = torch.cat([idx_unlabeled, idx_unlabeled], dim=0)

graphsage = GSGCN(opt, inputs, adj_lists)
trainer = Trainer(opt, graphsage)

def compute_f1(true, pred):
    true = true.data.numpy()
    pred = pred.data.cpu().numpy()

    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0

    return f1_score(true, pred, average="micro")

def get_current_consistency_weight(final_consistency_weight, epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    epoch = epoch - args.consistency_rampup_starts
    #epoch = epoch + step_in_epoch / total_steps_in_epoch
    return final_consistency_weight *sigmoid_rampup(epoch, args.consistency_rampup_ends - args.consistency_rampup_starts )

results = []
for epoch in range(opt['epoch']):

    bg = int(random.random() * idx_train.size(0))
    ed = bg + opt['batch_size']
    batch_nodes = idx_train_double[bg:ed]

    bg = int(random.random() * idx_unlabeled.size(0))
    ed = bg + opt['batch_size']
    batch_nodes_unlabeled = idx_unlabeled_double[bg:ed]

    graphsage.train()
    target_labeled = target[batch_nodes]
    '''
    k = 1
    temp  = torch.zeros([k, target_labeled.shape[0], target_labeled.shape[1]], dtype=target_labeled.dtype)
    temp = temp.cuda()
    for i in range(k):
        temp[i,:,:] = trainer.predict_noisy(batch_nodes_unlabeled)
    target_predict = temp.mean(dim = 0)

    '''

    loss = trainer.loss_gs(batch_nodes, target_labeled)
    #mixup_consistency = get_current_consistency_weight(opt['mixup_consistency'], epoch)
    #loss += mixup_consistency * trainer.loss_gs(batch_nodes_unlabeled, target_predict)
    trainer.optimizer.zero_grad()
    loss.backward()
    trainer.optimizer.step()



    #idx_train = idx_train[torch.randperm(idx_train.size(0))]
    #start_time = time.time()
    #loss = trainer.update_gs(batch_nodes, target)
    #end_time = time.time()
    #times.append(end_time-start_time)

    '''
    rand_index = random.randint(0,1)
    #rand_index = 1# if random.random() < 0.9 else 0
    if rand_index == 0: ## do the augmented node training

        target_labeled = target[batch_nodes]

        k = 10
        temp  = torch.zeros([k, target_labeled.shape[0], target_labeled.shape[1]], dtype=target_labeled.dtype)
        temp = temp.cuda()
        for i in range(k):
            temp[i,:,:] = trainer.predict_noisy(batch_nodes_unlabeled)
        target_predict = temp.mean(dim = 0)
        loss , loss_usup= trainer.update_soft_aux(batch_nodes, batch_nodes_unlabeled, target_labeled, target_predict, mixup_layer =[1])## for augmented nodes
        mixup_consistency = get_current_consistency_weight(opt['mixup_consistency'], epoch)
        total_loss = loss + mixup_consistency * loss_usup
        trainer.model.train()
        trainer.optimizer.zero_grad()
        total_loss.backward()
        trainer.optimizer.step()
    else:
        loss = trainer.update_gs(batch_nodes, target[batch_nodes])
    '''

    graphsage.eval()
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
