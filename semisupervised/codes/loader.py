import sys
import os
import math
import numpy as np
import torch
from torch.autograd import Variable

class Vocab(object):

    def __init__(self, file_name, cols, with_padding=False):
        self.itos = []
        self.stoi = {}
        self.vocab_size = 0

        if with_padding:
            string = '<pad>'
            self.stoi[string] = self.vocab_size
            self.itos.append(string)
            self.vocab_size += 1

        fi = open(file_name, 'r')
        for line in fi:
            items = line.strip().split('\t')
            for col in cols:
                item = items[col]
                strings = item.strip().split(' ')
                for string in strings:
                    if string not in self.stoi:
                        self.stoi[string] = self.vocab_size
                        self.itos.append(string)
                        self.vocab_size += 1
        fi.close()

    def __len__(self):
        return self.vocab_size

class EntityLabel(object):

    def __init__(self, file_name, entity, label):
        self.vocab_n, self.col_n = entity
        self.vocab_l, self.col_l = label
        self.itol = [-1 for k in range(self.vocab_n.vocab_size)]

        fi = open(file_name, 'r')
        for line in fi:
            items = line.strip().split('\t')
            sn, sl = items[self.col_n], items[self.col_l]
            n = self.vocab_n.stoi.get(sn, -1)
            l = self.vocab_l.stoi.get(sl, -1)
            if n == -1:
                continue
            self.itol[n] = l
        fi.close()

class EntityFeature(object):

    def __init__(self, file_name, entity, feature):
        self.vocab_n, self.col_n = entity
        self.vocab_f, self.col_f = feature
        self.itof = [[] for k in range(len(self.vocab_n))]
        self.one_hot = []

        fi = open(file_name, 'r')
        for line in fi:
            items = line.strip().split('\t')
            sn, sf = items[self.col_n], items[self.col_f]
            n = self.vocab_n.stoi.get(sn, -1)
            if n == -1:
                continue
            for s in sf.strip().split(' '):
                f = self.vocab_f.stoi.get(s, -1)
                if f == -1:
                    continue
                self.itof[n].append(f)
        fi.close()

    def to_one_hot(self):
        self.one_hot = [[0 for j in range(len(self.vocab_f))] for i in range(len(self.vocab_n))]
        for k in range(len(self.vocab_n)):
            for fid in self.itof[k]:
                self.one_hot[k][fid] = 1.0 / len(self.itof[k])

class Graph(object):
    def __init__(self):
        self.edges = []

        self.node_size = -1

        self.eid2iid = None
        self.iid2eid = None

        self.adj_w = None
        self.adj_t = None

    def add_edge(self, edge):
        self.edges.append(edge)

    def get_node_size(self):
        return self.node_size

    def get_edge_size(self):
        return len(self.edges)

    def to_symmetric(self, self_link_weight=1.0):
        #self.edges = [(u, v, w, t) for u, v, w, t in self.edges] + [(v, u, w, t) for u, v, w, t in self.edges]

        vocab = set()
        for u, v, w, t in self.edges:
            vocab.add(u)
            vocab.add(v)

        pair2wt = dict()
        for u, v, w, t in self.edges:
            pair2wt[(u, v)] = w

        edges_ = list()
        for (u, v), w in pair2wt.items():
            if u == v:
                continue
            w_ = pair2wt.get((v, u), -1)
            if w > w_:
                edges_ += [(u, v, w, 0), (v, u, w, 0)]
            elif w == w_:
                edges_ += [(u, v, w, 0)]
        for k in vocab:
            edges_ += [(k, k, self_link_weight, 0)]
        
        d = dict()
        for u, v, w, t in edges_:
            d[u] = d.get(u, 0.0) + w

        self.edges = [(u, v, w/math.sqrt(d[u]*d[v]), t) for u, v, w, t in edges_]

    def build_internal_vocab(self):
        self.eid2iid = dict()
        self.iid2eid = list()
        self.node_size = 0

        for u, v, w, t in self.edges:
            if u not in self.eid2iid:
                self.eid2iid[u] = self.node_size
                self.iid2eid.append(u)
                self.node_size += 1
            if v not in self.eid2iid:
                self.eid2iid[v] = self.node_size
                self.iid2eid.append(v)
                self.node_size += 1

    def build_adjacency(self):
        if self.node_size == -1:
            self.build_internal_vocab()

        self.adj_w = np.zeros([self.node_size, self.node_size], float)
        self.adj_t = np.zeros([self.node_size, self.node_size], int)

        for u, v, w, t in self.edges:
            iu = self.eid2iid[u]
            iv = self.eid2iid[v]

            self.adj_w[iu, iv] = w
            self.adj_t[iu, iv] = t

    def pad_adjacency(self, size):
        if self.adj_w is None or self.adj_t is None:
            self.build_adjacency()

        assert size >= self.node_size

        vb = self.iid2eid + [0 for k in range(self.node_size, size)]
        vb = np.array(vb)

        aw = np.zeros([size, size], float)
        aw[0:self.node_size, 0:self.node_size] = self.adj_w

        at = np.zeros([size, size], int)
        at[0:self.node_size, 0:self.node_size] = self.adj_t

        return vb, aw, at

    def get_slices(self, vocab_size, cuda=True):
        st = set()
        for u, v, w, t in self.edges:
            st.add(t)

        shape = torch.Size([vocab_size, vocab_size])

        slices = []
        for t_ in st:
            us, vs, ws = [], [], []
            for u, v, w, t in self.edges:
                if t != t_:
                    continue
                us += [u]
                vs += [v]
                ws += [w]
            index = torch.LongTensor([us, vs])
            value = torch.Tensor(ws)
            if cuda:
                index = index.cuda()
                value = value.cuda()
            adj = torch.sparse.FloatTensor(index, value, shape)
            if cuda:
                adj = adj.cuda()
            slices += [adj]

        return slices

class Graphs(object):

    def __init__(self, file_name, entity, weight=None, etype=None, graph=None):
        self.vocab_n, self.col_u, self.col_v = entity
        self.col_w = weight
        self.vocab_t, self.col_t = None, None
        if etype != None:
            self.vocab_t, self.col_t = etype
        self.vocab_g, self.col_g = None, None
        if graph != None:
            self.vocab_g, self.col_g = graph

        self.graph_size = 1
        if self.vocab_g != None:
            self.graph_size = len(self.vocab_g)
        self.graph = [Graph() for k in range(self.graph_size)]

        fi = open(file_name, 'r')
        for line in fi:
            items = line.strip().split('\t')

            su, sv = items[self.col_u], items[self.col_v]
            sw = items[self.col_w] if self.col_w != None else None
            st = items[self.col_t] if self.col_t != None else None
            sg = items[self.col_g] if self.col_g != None else None

            u, v = self.vocab_n.stoi.get(su, -1), self.vocab_n.stoi.get(sv, -1)
            w = float(sw) if sw != None else 1
            t = self.vocab_t.stoi.get(st, -1) if st != None else 0
            g = self.vocab_g.stoi.get(sg, -1) if sg != None else 0

            if u == -1 or v == -1 or w <= 0 or t == -1 or g == -1:
                continue

            self.graph[g].add_edge((u, v, w, t))
        fi.close()

    def __len__(self):
        return self.graph_size

    def __getitem__(self, idx):
    	return self.graph[idx]

    def to_symmetric(self, self_link_weight=1.0):
    	for graph in self.graph:
    		graph.to_symmetric(self_link_weight)

    def build_adjacency(self):
    	for graph in self.graph:
    		graph.build_adjacency()

    def get_batch(self, indices):
        max_l = max([self.graph[i].get_node_size() for i in indices])

        vbs, aws, ats = [], [], []
        for i in indices:
            vb, aw, at = self.graph[i].pad_adjacency(max_l)

            vb = vb.reshape([1, max_l])
            aw = aw.reshape([1, max_l, max_l])
            at = at.reshape([1, max_l, max_l])

            vbs += [vb]
            aws += [aw]
            ats += [at]

        vbs = np.concatenate(vbs, axis=0)
        aws = np.concatenate(aws, axis=0)
        ats = np.concatenate(ats, axis=0)

        vbs = Variable(torch.LongTensor(vbs))
        aws = Variable(torch.Tensor(aws))
        ats = Variable(torch.LongTensor(ats))

        return vbs, aws, ats

    def get_slices(self, index=0, cuda=True):
        return self.graph[index].get_slices(self.vocab_n.vocab_size, cuda)

    def self_links(self, index=0):
    	for k in range(self.vocab_n.vocab_size):
    		self.graph[index].add_edge((k, k, 1, 0))