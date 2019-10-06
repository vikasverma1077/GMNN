import math
import random
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

class mean_aggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, opt, emb): 
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(mean_aggregator, self).__init__()
        self.opt = opt
        self.emb = emb
        
    def forward(self, nodes, neighbors, num_sample=5):
        """
        nodes --- list of nodes in a batch
        neighbors --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in neighbors]
        else:
            samp_neighs = neighbors

        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.opt['cuda']:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        if self.opt['cuda']:
            embed_matrix = self.emb(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.emb(torch.LongTensor(unique_nodes_list))
        to_feats = mask.mm(embed_matrix)
        return to_feats

class graphsage_encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, opt, emb, neighbors, aggregator, base_model=None): 
        super(graphsage_encoder, self).__init__()
        self.opt = opt
        self.emb = emb
        self.neighbors = neighbors
        self.aggregator = aggregator
        if base_model != None:
            self.base_model = base_model
        self.in_size = opt['in']
        self.out_size = opt['out']

        self.weight = nn.Parameter(torch.FloatTensor(self.out_size, self.in_size if self.opt['gcn'] != 0 else 2 * self.in_size))
        nn.init.xavier_uniform_(self.weight)

        if opt['cuda']:
            self.cuda()

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        neigh_feats = self.aggregator.forward(nodes, [self.neighbors[int(node)] for node in nodes], num_sample=self.opt['num_sample'])
        if self.opt['gcn'] == 0:
            if self.opt['cuda']:
                self_feats = self.emb(nodes.cuda())
            else:
                self_feats = self.emb(nodes)
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        combined = F.relu(self.weight.mm(combined.t()))
        return combined
