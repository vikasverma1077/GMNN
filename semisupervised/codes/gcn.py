import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from layer import message
import random


def mixup_gnn_hidden(x, target, train_idx, alpha):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)#, size = train_idx.shape[0])#+0.001
    else:
        lam = 1.
    permuted_train_idx = train_idx[torch.randperm(train_idx.shape[0])]
    x[train_idx] = lam*x[train_idx]+ (1-lam)*x[permuted_train_idx]
    return x, target[train_idx], target[permuted_train_idx],lam



class GCN(nn.Module):
    def __init__(self, opt, graphs):
        super(GCN, self).__init__()
        self.opt = opt
        self.graphs = graphs

        opt_ = dict([('types', 1), ('in', opt['num_feature']), ('out', opt['hidden_dim'])])
        self.m1 = message(opt_, graphs)

        opt_ = dict([('types', 1), ('in', opt['hidden_dim']), ('out', opt['num_class'])])
        self.m2 = message(opt_, graphs)

        if opt['cuda']:
            self.cuda()

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        
        x = self.m1(x)
        x = F.relu(x)
        
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        
        x = self.m2(x)
        return x
    
    
    def forward_aux(self, x, target=None, train_idx= None, mixup_input= False, mixup_hidden = False, mixup_alpha = 0.0,layer_mix=None):

        if mixup_hidden == True or mixup_input == True:
            if mixup_hidden == True:
                layer_mix = random.choice(layer_mix)
            elif mixup_input == True:
                layer_mix = 0


            if layer_mix ==0:
                x, target_a, target_b, lam = mixup_gnn_hidden(x, target, train_idx, mixup_alpha)

            x = F.dropout(x, self.opt['input_dropout'], training=self.training)

            x = self.m1.forward_aux(x)
            x = F.relu(x)
            if layer_mix == 1:
                x, target_a, target_b, lam = mixup_gnn_hidden(x, target, train_idx, mixup_alpha)

            x = F.dropout(x, self.opt['dropout'], training=self.training)
            x = self.m2.forward_aux(x)

            return x, target_a, target_b, lam

        else:

            x = F.dropout(x, self.opt['input_dropout'], training=self.training)
            x = self.m1.forward_aux(x)
            x = F.relu(x)
            x = F.dropout(x, self.opt['dropout'], training=self.training)
            x = self.m2.forward_aux(x)
            return x
