import math
import numpy as np
import random
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from layer import GraphConvolution

def mixup_data(x, y, alpha):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    #y_a, y_b = y, y[index]
    mixed_y = lam * y + (1 - lam) * y[index,:]
    return mixed_x, mixed_y


class GNN_mix(nn.Module):
    def __init__(self, opt, adj):
        super(GNNq, self).__init__()
        self.opt = opt
        self.adj = adj

        opt_ = dict([('in', opt['num_feature']), ('out', opt['hidden_dim'])])
        self.m1 = GraphConvolution(opt_, adj)

        opt_ = dict([('in', opt['hidden_dim']), ('out', opt['num_class'])])
        self.m2 = GraphConvolution(opt_, adj)

        if opt['cuda']:
            self.cuda()

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def forward(self, x, target=None, train_idx= None, mix=False):
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        if mix == True:
            x_labeled = x[train_idx]
            y_labeled = target[train_idx]
        x = self.m1(x)
        x = F.relu(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = self.m2(x)
        return x

class GNNq(nn.Module):
    def __init__(self, opt, adj):
        super(GNNq, self).__init__()
        self.opt = opt
        self.adj = adj

        opt_ = dict([('in', opt['num_feature']), ('out', opt['hidden_dim'])])
        self.m1 = GraphConvolution(opt_, adj)

        opt_ = dict([('in', opt['hidden_dim']), ('out', opt['num_class'])])
        self.m2 = GraphConvolution(opt_, adj)

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

class GNNp(nn.Module):
    def __init__(self, opt, adj):
        super(GNNp, self).__init__()
        self.opt = opt
        self.adj = adj

        opt_ = dict([('in', opt['num_class']), ('out', opt['hidden_dim'])])
        self.m1 = GraphConvolution(opt_, adj)

        opt_ = dict([('in', opt['hidden_dim']), ('out', opt['num_class'])])
        self.m2 = GraphConvolution(opt_, adj)

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
    

class MLP(nn.Module):
    def __init__(self, opt):
        super(MLP, self).__init__()
        self.opt = opt
        self.fc1 = nn.Linear(opt['num_feature'],500) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(500, 250) 
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(250, 100)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        self.fc5 = nn.Linear(50, opt['hidden_dim'])
        self.relu = nn.ReLU()
        self.fc6 = nn.Linear(opt['hidden_dim'],opt['num_class'] )
        
        if opt['cuda']:
            self.cuda()
        
    
    def forward(self, x, target=None, mixup_input = False, mixup_hidden = False,  mixup_alpha = 0.1, layer_mix=None):
        if mixup_hidden == True:
            layer_mix = random.randint(1,layer_mix)
        elif mixup_input == True:
            layer_mix = 0
        
        out = x
        if layer_mix == 0:
            out, mixed_target = mixup_data(out, target, mixup_alpha)
        out = self.fc1(x)
        out = self.relu(out)
        
        if layer_mix == 1:
            out, mixed_target = mixup_data(out, target, mixup_alpha)
        out = self.fc2(out)
        out = self.relu(out)
        if layer_mix == 2:
            out, mixed_target = mixup_data(out, target, mixup_alpha)
        out = self.fc3(out)
        out = self.relu(out)
        if layer_mix == 3:
            out, mixed_target = mixup_data(out, target, mixup_alpha)
        out = self.fc4(out)
        out = self.relu(out)
        if layer_mix == 4:
            out, mixed_target = mixup_data(out, target, mixup_alpha)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.fc6(out)
        if layer_mix == None:
            return out
        else:
            return out, mixed_target


        
