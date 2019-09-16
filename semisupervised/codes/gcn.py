import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from layer import message

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