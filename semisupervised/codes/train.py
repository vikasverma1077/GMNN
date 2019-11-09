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

from ramps import *
from losses import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='data')
parser.add_argument('--save', type=str, default='/')
parser.add_argument('--hidden_dim', type=int, default=128, help='RNN hidden state size.')
parser.add_argument('--input_dropout', type=float, default=0.5, help='Input and RNN dropout rate.')
parser.add_argument('--dropout', type=float, default=0.0, help='Input and RNN dropout rate.')
parser.add_argument('--optimizer', type=str, default='adam', help='Optional info for the experiment.')
parser.add_argument('--mixup_alpha', type=float, default=0.1, help='alpha for mixing')
parser.add_argument('--lr', type=float, default=0.01, help='Applies to SGD and Adagrad.')
parser.add_argument('--decay', type=float, default=0)
parser.add_argument('--self_link_weight', type=float, default=1.0)
parser.add_argument('--pre_epoch', type=int, default=150)
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

parser.add_argument('--ema_decay', default=0.999, type=float, metavar='ALPHA',
                    help='ema variable decay rate (default: 0.999)')
#parser.add_argument('--consistency', default=None, type=float, metavar='WEIGHT',
#                    help='use consistency loss with given weight (default: None)')
parser.add_argument('--consistency_type', default="mse", type=str, metavar='TYPE',
                     choices=['mse', 'kl'],
                     help='consistency loss type to use')
parser.add_argument('--consistency_rampup_starts', default=75, type=int, metavar='EPOCHS',
                     help='epoch at which consistency loss ramp-up starts')
parser.add_argument('--consistency_rampup_ends', default=150, type=int, metavar='EPOCHS',
                     help='lepoch at which consistency loss ramp-up ends')
#parser.add_argument('--mixup_sup_alpha', default=0.0, type=float,
#                    help='for supervised loss, the alpha parameter for the beta distribution from where the mixing lambda is drawn')
#parser.add_argument('--mixup_usup_alpha', default=0.0, type=float,
#                    help='for unsupervised loss, the alpha parameter for the beta distribution from where the mixing lambda is drawn')
#parser.add_argument('--mixup_hidden', action='store_true',
#                    help='apply mixup in hidden layers')
#parser.add_argument('--num_mix_layer', default=3, type=int,
#                    help='number of hidden layers on which mixup is applied in addition to input layer')
parser.add_argument('--mixup_consistency', default=0.1, type=float,
                     help='max consistency coeff for mixup usup loss')



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
idx_unlabeled = list(set(idx_all)-set(idx_train))


# model
# initialize a relation model
if opt['cuda']:
    slices = [s.cuda() for s in slices]

gcn = GCN(opt, slices)
trainer_gcn = Trainer(opt, gcn)

gcn_ema = GCN(opt, slices)

for ema_param, param in zip(gcn_ema.parameters(), gcn.parameters()):
    ema_param.data= param.data

for param in gcn_ema.parameters():
    param.detach_()

trainer_gcn_ema = Trainer(opt, gcn_ema, ema = True)

inputs = torch.Tensor(feature.one_hot)
target = torch.LongTensor(label.itol)
idx_train = torch.LongTensor(idx_train)
idx_dev = torch.LongTensor(idx_dev)
idx_test = torch.LongTensor(idx_test)
idx_all = torch.LongTensor(idx_all)
idx_unlabeled = torch.LongTensor(idx_unlabeled)
#inputs_q = torch.zeros(opt['num_node'], opt['num_feature'])
#target_q = torch.zeros(opt['num_node'], opt['num_class'])
#inputs_p = torch.zeros(opt['num_node'], opt['num_class'])
#target_p = torch.zeros(opt['num_node'], opt['num_class'])

if opt['cuda']:
    inputs = inputs.cuda()
    target = target.cuda()
    idx_train = idx_train.cuda()
    idx_dev = idx_dev.cuda()
    idx_test = idx_test.cuda()
    idx_all = idx_all.cuda()
    #inputs_q = inputs_q.cuda()
    #target_q = target_q.cuda()
    #inputs_p = inputs_p.cuda()
    #target_p = target_p.cuda()
    

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


def update_ema_variables(model, ema_model, alpha, epoch):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (epoch + 1), alpha)
    #print (alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def get_current_consistency_weight(final_consistency_weight, epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    epoch = epoch - args.consistency_rampup_starts
    #epoch = epoch + step_in_epoch / total_steps_in_epoch
    return final_consistency_weight *sigmoid_rampup(epoch, args.consistency_rampup_ends - args.consistency_rampup_starts )



def sharpen(prob, temperature):
    temp_reciprocal = 1.0/ temperature
    prob = torch.pow(prob, temp_reciprocal)
    row_sum = prob.sum(dim=1).reshape(-1,1)
    out = prob/row_sum
    return out



def pre_train(epoches):
    init_gcn_data()

    results = []
    
    if args.consistency_type == 'mse':
        consistency_criterion = softmax_mse_loss # remember to divide by the batch size
    elif args.consistency_type == 'kl':
        consistency_criterion = softmax_kl_loss
    
    for epoch in range(epoches):
        rand_index = random.randint(0,1)
        if rand_index == 0: ## do the augmented node training
    	    trainer_gcn.model.train()
            trainer_gcn.optimizer.zero_grad()
            ## get the psudolabels for the unlabeled nodes ##
            #import pdb; pdb.set_trace()
                
            k = 10
            temp  = torch.zeros([k, target_gcn.shape[0], target_gcn.shape[1]], dtype=target_gcn.dtype)
            temp = temp.cuda()
            for i in range(k):
                temp[i,:,:] = trainer_gcn.predict_noisy(inputs_gcn)
            target_predict = temp.mean(dim = 0)# trainer_q.predict(inputs_q)
                
            #target_predict = trainer_gcn.predict(inputs_gcn)
            target_predict = sharpen(target_predict,0.1)
            #if epoch == 500:
            #    print (target_predict)
            target_gcn[idx_unlabeled] = target_predict[idx_unlabeled]
            #inputs_q_new, target_q_new, idx_train_new = get_augmented_network_input(inputs_q, target_q,idx_train,opt, net_file, net_temp_file) ## get the augmented nodes in the input space
            #idx_train_new = 
            #loss = trainer_q.update_soft_mix(inputs_q, target_q, idx_train)## for mixing features
            temp = torch.randint(0, idx_unlabeled.shape[0], size=(idx_train.shape[0],))## index of the samples chosen from idx_unlabeled
            idx_unlabeled_subset = idx_unlabeled[temp]
            loss , loss_usup= trainer_gcn.update_soft_aux(inputs_gcn, target_gcn, target, idx_train, idx_unlabeled_subset, opt, mixup_layer =[1])## for augmented nodes
            mixup_consistency = get_current_consistency_weight(opt['mixup_consistency'], epoch)
            total_loss = loss + mixup_consistency*loss_usup
            #trainer_gcn.model.train()
            #trainer_gcn.optimizer.zero_grad()
            total_loss.backward()
            trainer_gcn.optimizer.step()

        else:
            trainer_gcn.model.train()
            trainer_gcn.optimizer.zero_grad()
            loss = trainer_gcn.update_soft(inputs_gcn, target_gcn, idx_train)
            
            k = 10
            temp  = torch.zeros([k, target_gcn.shape[0], target_gcn.shape[1]], dtype=target_gcn.dtype)
            temp = temp.cuda()
            for i in range(k):
                temp[i,:,:] = trainer_gcn.predict_noisy(inputs_gcn)
            target_predict = temp.mean(dim = 0)# trainer_q.predict(inputs_q)
            target_predict = sharpen(target_predict,0.1)
            target_q[idx_unlabeled] = target_predict[idx_unlabeled]
            
            temp = torch.randint(0, idx_unlabeled.shape[0], size=(idx_train.shape[0],))## index of the samples chosen from idx_unlabeled
            idx_unlabeled_subset = idx_unlabeled[temp]
            
            loss_usup = trainer_gcn.update_soft(inputs_gcn, target_gcn, idx_unlabeled_subset)

            mixup_consistency = get_current_consistency_weight(opt['mixup_consistency'], epoch)
            total_loss = loss + mixup_consistency*loss_usup
            
            #total_loss = loss
            #trainer_gcn.model.train()
            #trainer_gcn.optimizer.zero_grad()
            total_loss.backward()
            trainer_gcn.optimizer.step()

        #loss = trainer_gcn.update_soft(inputs_gcn, target_gcn, idx_train)

        _, preds, accuracy_dev = trainer_gcn.evaluate(inputs_gcn, target, idx_dev)
        f1_dev = f1_score(target[idx_dev].cpu().numpy(), preds.cpu().numpy(), average='macro')

        _, preds, accuracy_test = trainer_gcn.evaluate(inputs_gcn, target, idx_test)
        f1_test = f1_score(target[idx_test].cpu().numpy(), preds.cpu().numpy(), average='macro')

        results += [(f1_dev, f1_test)]
        print(epoch, f1_dev, f1_test)


        #update_ema_variables(gcn, gcn_ema, opt['ema_decay'], epoch)

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

