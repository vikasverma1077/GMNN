import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

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

net_file = 'cora/net.txt'
label_file = 'cora/label.txt'
feature_file = 'cora/feature.txt'

vocab_node = loader.Vocab(net_file, [0, 1])
vocab_label = loader.Vocab(label_file, [1])
vocab_feature = loader.Vocab(feature_file, [1])

graph = loader.Graphs(file_name=net_file, entity=[vocab_node, 0, 1])
label = loader.EntityLabel(file_name=label_file, entity=[vocab_node, 0], label=[vocab_label, 1])
feature = loader.EntityFeature(file_name=feature_file, entity=[vocab_node, 0], feature=[vocab_feature, 1])

graph.build_neighbor()
feature.to_one_hot(normalize=False)

feat_data = np.array(feature.one_hot)
labels = np.array([[l] for l in label.itol])
adj_lists = graph[0].nbs


opt = dict()
opt['num_node'] = len(vocab_node)
opt['num_feature'] = len(vocab_feature)
opt['num_class'] = len(vocab_label)
opt['hidden_dim'] = 128
opt['gcn'] = 1
opt['cuda'] = True
opt['num_sample'] = 5
opt['optimizer'] = 'sgd'
opt['lr'] = 0.7
opt['decay'] = 0.0

np.random.seed(1)
random.seed(1)
num_nodes = 2708

features = nn.Embedding(2708, 1432)
features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
features.cuda()

graphsage = GSGCN(opt, feat_data, adj_lists)
trainer = Trainer(opt, graphsage)
rand_indices = np.random.permutation(num_nodes)
test = rand_indices[:1000]
val = rand_indices[1000:1500]
train = list(rand_indices[1500:])

optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
times = []
for batch in range(100):
    batch_nodes = train[:256]
    random.shuffle(train)
    start_time = time.time()
    loss = trainer.update_gs(batch_nodes, torch.LongTensor(label.itol))
    end_time = time.time()
    times.append(end_time-start_time)
    print(batch, loss)

val_output = graphsage.forward(val) 
print("Validation F1:", f1_score(labels[val], val_output.data.cpu().numpy().argmax(axis=1), average="micro"))
print("Average batch time:", np.mean(times))
