import sys
import os
import copy
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

from trainer import Trainer
from gnn import GNNq, GNNp, MLP, GNN_mix
import loader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='data')
parser.add_argument('--save', type=str, default='exp', help = 'name of the folder where the results are saved')
parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
parser.add_argument('--mixup_alpha', type=float, default=1.0, help='alpha for mixing')
parser.add_argument('--self_link_weight', type=float, default=1.0, help='Weight of self-links.')
parser.add_argument('--pre_epoch', type=int, default=200, help='Number of pre-training epochs.')
parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs per iteration.')
parser.add_argument('--iter', type=int, default=10, help='Number of training iterations.')
parser.add_argument('--use_gold', type=int, default=1, help='Whether using the ground-truth label of labeled objects, 1 for using, 0 for not using.')
parser.add_argument('--tau', type=float, default=1.0, help='Annealing temperature in sampling.')
parser.add_argument('--draw', type=str, default='max', help='Method for drawing object labels, max for max-pooling, smp for sampling.')
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

#import pdb;pdb.set_trace()
### create a temporart net file
exp_dir = os.path.join(os.getcwd(),args.save)
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
net_temp_file = os.path.join(exp_dir,'net_temp.txt')

#net_file = exp_dir+ '/net_temp.txt'

vocab_node = loader.Vocab(net_file, [0, 1])
vocab_label = loader.Vocab(label_file, [1])
vocab_feature = loader.Vocab(feature_file, [1])

opt['num_node'] = len(vocab_node)
opt['num_feature'] = len(vocab_feature)
opt['num_class'] = len(vocab_label)


graph = loader.Graph(file_name=net_file, entity=[vocab_node, 0, 1])
label = loader.EntityLabel(file_name=label_file, entity=[vocab_node, 0], label=[vocab_label, 1])
feature = loader.EntityFeature(file_name=feature_file, entity=[vocab_node, 0], feature=[vocab_feature, 1])
#import pdb; pdb.set_trace()
graph.to_symmetric(opt['self_link_weight'])
feature.to_one_hot(binary=True)
adj = graph.get_sparse_adjacency(opt['cuda'])

with open(train_file, 'r') as fi:
    idx_train = [vocab_node.stoi[line.strip()] for line in fi]
with open(dev_file, 'r') as fi:
    idx_dev = [vocab_node.stoi[line.strip()] for line in fi]
with open(test_file, 'r') as fi:
    idx_test = [vocab_node.stoi[line.strip()] for line in fi]
idx_all = list(range(opt['num_node']))

inputs = torch.Tensor(feature.one_hot)
target = torch.LongTensor(label.itol)
idx_train = torch.LongTensor(idx_train)
idx_dev = torch.LongTensor(idx_dev)
idx_test = torch.LongTensor(idx_test)
idx_all = torch.LongTensor(idx_all)
inputs_q = torch.zeros(opt['num_node'], opt['num_feature'])
target_q = torch.zeros(opt['num_node'], opt['num_class'])
inputs_p = torch.zeros(opt['num_node'], opt['num_class'])
target_p = torch.zeros(opt['num_node'], opt['num_class'])

if opt['cuda']:
    inputs = inputs.cuda()
    target = target.cuda()
    idx_train = idx_train.cuda()
    idx_dev = idx_dev.cuda()
    idx_test = idx_test.cuda()
    idx_all = idx_all.cuda()
    inputs_q = inputs_q.cuda()
    target_q = target_q.cuda()
    inputs_p = inputs_p.cuda()
    target_p = target_p.cuda()

gnnq = GNNq(opt, adj)
#gnnq = MLP(opt)
#gnnq = GNN_mix(opt, adj)
trainer_q = Trainer(opt, gnnq)

gnnp = GNNp(opt, adj)
trainer_p = Trainer(opt, gnnp)

def init_q_data():
    inputs_q.copy_(inputs)
    temp = torch.zeros(idx_train.size(0), target_q.size(1)).type_as(target_q)
    temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
    target_q[idx_train] = temp

def update_p_data():
    preds = trainer_q.predict(inputs_q, opt['tau'])
    if opt['draw'] == 'exp':
        inputs_p.copy_(preds)
        target_p.copy_(preds)
    elif opt['draw'] == 'max':
        idx_lb = torch.max(preds, dim=-1)[1]
        inputs_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
        target_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
    elif opt['draw'] == 'smp':
        idx_lb = torch.multinomial(preds, 1).squeeze(1)
        inputs_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
        target_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
    if opt['use_gold'] == 1:
        temp = torch.zeros(idx_train.size(0), target_q.size(1)).type_as(target_q)
        temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
        inputs_p[idx_train] = temp
        target_p[idx_train] = temp

def update_q_data():
    preds = trainer_p.predict(inputs_p)
    target_q.copy_(preds)
    if opt['use_gold'] == 1:
        temp = torch.zeros(idx_train.size(0), target_q.size(1)).type_as(target_q)
        temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
        target_q[idx_train] = temp

def pre_train(epoches):
    best = 0.0
    init_q_data()
    results = []
    for epoch in range(epoches):
        #loss = trainer_q.update_soft_mlp(inputs_q, target_q, idx_train)
        #loss = trainer_q.update_soft(inputs_q, target_q, idx_train)
        #import pdb; pdb.set_trace()
        ### create mix of feature and labels
        rand_index = 0 #random.randint(0,1)
        #print (rand_index)
        if epoch > 1000:
            ### create a new net file###
            if os.path.exists(net_temp_file):
                os.remove(net_temp_file)
                copyfile(net_file, net_temp_file)
            else:
                copyfile(net_file, net_temp_file)
            #import pdb; pdb.set_trace()
            lamb = np.random.beta(opt['mixup_alpha'],opt['mixup_alpha'], size = target_q[idx_train].shape[0])
            lamb = torch.from_numpy(lamb)
            lamb = lamb.reshape(-1,1).type(torch.FloatTensor).cuda()

            inputs_q_new = inputs_q
            target_q_new = target_q
            idx_train_new = torch.tensor([], dtype= idx_train.dtype).cuda()# idx_train# [] for not adding the original idx_train in the additional train data
            target_new = target
    
            for j in range(2):
                #import pdb; pdb.set_trace()
                permuted_train_idx = idx_train[torch.randperm(idx_train.shape[0])]
                train_x_additional = lamb.repeat(1,inputs_q[idx_train].shape[1])*inputs_q[idx_train]+ (1-lamb).repeat(1,inputs_q[idx_train].shape[1])*inputs_q[permuted_train_idx]
                
                ###row wise normalization : the sum of the row should be one
                
                train_y_additional = lamb.repeat(1,target_q[idx_train].shape[1])*target_q[idx_train]+ (1-lamb).repeat(1,target_q[idx_train].shape[1])*target_q[permuted_train_idx]
                idx_train_additional = np.arange(idx_train.shape[0])
                idx_train_additional = torch.from_numpy(idx_train_additional)
                idx_train_additional = idx_train_additional.cuda()
                idx_train_additional = idx_train_additional + target_q_new.shape[0]
    
                inputs_q_new = torch.cat((inputs_q_new, train_x_additional),0)
                target_q_new = torch.cat((target_q_new, train_y_additional),0)
                idx_train_new = torch.cat((idx_train_new, idx_train_additional),0)
            
                ## add dummy labels to the target tensor, these dummy values will not be used so I just used '0'##
                #import pdb; pdb.set_trace()
                
                temp = torch.zeros(train_y_additional.shape[0], dtype = target.dtype)
                temp = temp.cuda()
                target_new = torch.cat((target_new, temp),0)
    
                #import pdb; pdb.set_trace()
                fi = open(net_temp_file, 'a+')
                start_index_for_additional_nodes = target_q.shape[0]+j*idx_train.shape[0]
                for i in range(idx_train.shape[0]):
                    node_index = start_index_for_additional_nodes+i
                    fi.write(str(node_index)+'\t'+str(idx_train[i].item())+'\t'+str(1)+'\n')
                    fi.write(str(idx_train[i].item())+'\t'+str(node_index)+'\t'+str(1)+'\n')
                    fi.write(str(node_index)+'\t'+str(permuted_train_idx[i].item())+'\t'+str(1)+'\n')
                    fi.write(str(permuted_train_idx[i].item())+'\t'+str(node_index)+'\t'+str(1)+'\n')
                fi.close()
            
            #import pdb; pdb.set_trace()
            ## reload the net file in the adjacency matrix###
            vocab_node = loader.Vocab(net_temp_file, [0, 1])
            graph = loader.Graph(file_name=net_file, entity=[vocab_node, 0, 1])
            graph.to_symmetric(opt['self_link_weight'])
            adj_new = graph.get_sparse_adjacency(opt['cuda'])
            trainer_q.model.adj = adj_new
            trainer_q.model.m1.adj = adj_new
            trainer_q.model.m2.adj = adj_new
            #trainer_q.model.m3.adj = adj
            #trainer_q.model.m4.adj = adj
            
            #idx_train_new = 
            #loss = trainer_q.update_soft_mix(inputs_q, target_q, idx_train)## for mixing features 
            loss = trainer_q.update_soft(inputs_q_new, target_q_new, idx_train_new)## for augmented nodes
        else:
            loss = trainer_q.update_soft(inputs_q, target_q, idx_train)
        #loss = trainer_q.update_soft_aux(inputs_q, target_q, idx_train)## for training aux networks
        loss_aux = loss
        #loss, loss_aux = trainer_q.update_soft_aux(inputs_q, target_q, idx_train, epoch, opt)## for auxiliary net with shared parameter
        trainer_q.model.adj = adj
        trainer_q.model.m1.adj = adj
        trainer_q.model.m2.adj = adj
        _, preds, accuracy_train = trainer_q.evaluate(inputs_q, target, idx_train) ## target_new : for augmented nodes
        _, preds, accuracy_dev = trainer_q.evaluate(inputs_q, target, idx_dev)
        _, preds, accuracy_test = trainer_q.evaluate(inputs_q, target, idx_test)
        results += [(accuracy_dev, accuracy_test)]
        if epoch%100 == 0:
            print ( rand_index)
            print ('epoch :{:4d},loss:{:.15f},loss:{:.15f}, train_acc:{:.3f}, dev_acc:{:.3f}, test_acc:{:.3f}'.format(epoch, loss,loss_aux, accuracy_train, accuracy_dev, accuracy_test))

        if accuracy_dev > best:
            best = accuracy_dev
            state = dict([('model', copy.deepcopy(trainer_q.model.state_dict())), ('optim', copy.deepcopy(trainer_q.optimizer.state_dict()))])
    #trainer_q.model.load_state_dict(state['model'])
    #trainer_q.optimizer.load_state_dict(state['optim'])
    return results

def train_p(epoches):
    update_p_data()
    results = []
    for epoch in range(epoches):
        loss = trainer_p.update_soft(inputs_p, target_p, idx_all)
        _, preds, accuracy_dev = trainer_p.evaluate(inputs_p, target, idx_dev)
        _, preds, accuracy_test = trainer_p.evaluate(inputs_p, target, idx_test)
        results += [(accuracy_dev, accuracy_test)]
    return results

def train_q(epoches):
    update_q_data()
    results = []
    for epoch in range(epoches):
        loss = trainer_q.update_soft(inputs_q, target_q, idx_all)
        _, preds, accuracy_dev = trainer_q.evaluate(inputs_q, target, idx_dev)
        _, preds, accuracy_test = trainer_q.evaluate(inputs_q, target, idx_test)
        results += [(accuracy_dev, accuracy_test)]
    return results

base_results, q_results, p_results = [], [], []
base_results += pre_train(opt['pre_epoch'])

#for k in range(opt['iter']):
#    p_results += train_p(opt['epoch'])
#    q_results += train_q(opt['epoch'])

def get_accuracy(results):
    best_dev, acc_test = 0.0, 0.0
    for d, t in results:
        if d >= best_dev:
            best_dev, acc_test = d, t
    return acc_test

acc_test = get_accuracy(base_results)

print('Test acc{:.3f}'.format(acc_test * 100))

#if opt['save'] != '/':
#    trainer_q.save(opt['save'] + '/gnnq.pt')
#    trainer_p.save(opt['save'] + '/gnnp.pt')
##
