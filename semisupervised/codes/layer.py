import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class SparseMM(torch.autograd.Function):

    def __init__(self, sparse):
        super(SparseMM, self).__init__()
        self.sparse = sparse

    def forward(self, dense):
        return torch.mm(self.sparse, dense)

    def backward(self, grad_output):
        grad_input = None
        if self.needs_input_grad[0]:
            grad_input = torch.mm(self.sparse.t(), grad_output)
        return grad_input


class message(nn.Module):

    def __init__(self, opt, graphs):
        super(message, self).__init__()
        self.opt = opt

        self.tp_size = opt['types']
        self.in_size = opt['in']
        self.out_size = opt['out']

        self.graphs = graphs

        self.weight = Parameter(torch.Tensor(self.tp_size, self.in_size, self.out_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_size)
        self.weight.data.uniform_(-stdv, stdv)
    
    def forward(self, x):
        ms = []

        for k in range(self.tp_size):
            m_ = torch.mm(x, self.weight[k])
            m_ = SparseMM(self.graphs[k])(m_)
            m_ = m_.unsqueeze(0)
            ms += [m_]

        ms = torch.cat(ms, dim=0)
        ms = torch.sum(ms, dim=0)
        m = ms.squeeze(0)

        return m

    def forward_aux(self, x):
        ms = []

        for k in range(self.tp_size):
            m_ = torch.mm(x, self.weight[k])
            #m_ = SparseMM(self.graphs[k])(m_)
            m_ = m_.unsqueeze(0)
            ms += [m_]

            ms = torch.cat(ms, dim=0)
            ms = torch.sum(ms, dim=0)
            m = ms.squeeze(0)

            return m
