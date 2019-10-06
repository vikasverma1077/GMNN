import math
import random
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from graphsage.layer import message, mean_aggregator, graphsage_encoder

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

def mixup_gnn_hidden(x, target, train_idx, alpha):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)#, size = train_idx.shape[0])#+0.001
    else:
        lam = 1.
        #lam = np.repeat(lam, train_idx.shape[0])
    #import pdb; pdb.set_trace()
    #lam = lam.reshape(train_idx.shape[0],1)
    #lam = torch.from_numpy(lam).cuda().type(torch.float32)
    permuted_train_idx = torch.randperm(x.size(0))
    x = lam * x + (1-lam) * x[permuted_train_idx]
    #target[train_idx] = lam*target[train_idx]+ (1-lam)*target[permuted_train_idx]
    """
    ### mix the unlabeded nodes###
    all_idx = set(range(0,x.shape[0]))
    train_idx_set = set(list(train_idx.cpu().numpy()))
    unlabeled_idx = np.asarray(list(all_idx-train_idx_set))
    permuted_unlabeled_idx = unlabeled_idx[torch.randperm(unlabeled_idx.shape[0])]
    x[unlabeled_idx] = lam*x[unlabeled_idx]+ (1-lam)*x[permuted_unlabeled_idx]
    """
    return x, target, target[permuted_train_idx], lam

class GSGCN(nn.Module):
    def __init__(self, opt, features, neighbors):
        super(GSGCN, self).__init__()
        self.opt = opt
        
        self.features = nn.Embedding(opt['num_node'], opt['num_feature'])
        self.features.weight = nn.Parameter(torch.FloatTensor(features), requires_grad=False)
        self.features.cuda()

        self.neighbors = neighbors

        self.a1 = mean_aggregator(opt, self.features)

        opt_ = dict([('in', opt['num_feature']), ('out', opt['hidden_dim']), ('gcn', opt['gcn']), ('num_sample', 25), ('dropout', opt['dropout']), ('cuda', opt['cuda'])])
        self.m1 = graphsage_encoder(opt_, self.features, self.neighbors, self.a1)

        self.a2 = mean_aggregator(opt, lambda nodes:self.m1(nodes).t())

        opt_ = dict([('in', opt['hidden_dim']), ('out', opt['hidden_dim']), ('gcn', opt['gcn']), ('num_sample', 10), ('dropout', opt['dropout']), ('cuda', opt['cuda'])])
        self.m2 = graphsage_encoder(opt_, lambda nodes:self.m1(nodes).t(), self.neighbors, self.a2, self.m1)

        self.m3 = nn.Parameter(torch.FloatTensor(opt['num_class'], opt['hidden_dim']))
        nn.init.xavier_uniform_(self.m3)

        if opt['cuda']:
            self.cuda()

    def forward(self, nodes):
        x = self.m2(nodes)
        x = self.m3.mm(x)
        return x.t()

    def forward_aux(self, target=None, train_idx= None, mixup_input= False, mixup_hidden = False, mixup_alpha = 0.0,layer_mix=None):
        if self.opt['cuda']:
            x = self.features(train_idx.cuda())
        else:
            x = self.features(train_idx)

        if mixup_hidden == True or mixup_input == True:
            if mixup_hidden == True:
                layer_mix = random.choice(layer_mix)
            elif mixup_input == True:
                layer_mix = 0

            if layer_mix ==0:
                x, target_a, target_b, lam = mixup_gnn_hidden(x, target, train_idx, mixup_alpha)

            x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    
            if self.opt['gcn'] == 0:
                x = torch.cat([x, x], dim=1)
            else:
                x = x
            x = F.relu(self.m1.weight.mm(x.t()))
            
            if layer_mix == 1:
                x, target_a, target_b, lam = mixup_gnn_hidden(x, target, train_idx, mixup_alpha)

            if self.opt['gcn'] == 0:
                x = torch.cat([x, x], dim=1)
            else:
                x = x
            x = F.relu(self.m2.weight.mm(x.t()))

            x = self.m3.mm(x).t()
            
            return x, target_a, target_b, lam
        
        else:
        
            if self.opt['gcn'] == 0:
                x = torch.cat([x, x], dim=1)
            else:
                x = x

            x = F.relu(self.m1.weight.mm(x.t()))

            if self.opt['gcn'] == 0:
                x = torch.cat([x, x], dim=1)
            else:
                x = x
            x = F.relu(self.m2.weight.mm(x.t()))

            x = self.m3.mm(x).t()

            return x

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return F.cross_entropy(scores, labels.squeeze())
