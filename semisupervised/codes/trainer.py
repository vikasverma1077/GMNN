import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from gcn import GCN
import torch_utils

class Trainer(object):
    """ A wrapper class for the training and evaluation of models. """
    def __init__(self, opt, model):
        # options
        self.opt = opt
        # model
        self.model = model
        # loss function
        self.criterion = nn.CrossEntropyLoss()
        # all parameters of the model
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        # whether moving all data to gpu
        if opt['cuda']:
            self.criterion.cuda()
        # intialize the optimizer
        self.optimizer = torch_utils.get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])

    def reset(self):
        self.model.reset()
        # intialize the optimizer
        self.optimizer = torch_utils.get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])

    # train the model with a batch
    def update(self, inputs, target, idx):
        """ Run a step of forward and backward model update. """
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(inputs)
        loss = self.criterion(logits[idx], target[idx])
        
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # train the model with a batch
    def update_soft(self, inputs, target, idx):
        """ Run a step of forward and backward model update. """
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(inputs)
        logits = torch.log_softmax(logits, dim=-1)
        loss = -torch.mean(torch.sum(target[idx] * logits[idx], dim=-1))
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def evaluate(self, inputs, target, idx):
        """ Run a step of forward and backward model update. """
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.eval()

        logits = self.model(inputs)
        loss = self.criterion(logits[idx], target[idx])
        preds = torch.max(logits[idx], dim=1)[1]
        correct = preds.eq(target[idx]).double()
        accuracy = correct.sum() / idx.size(0)

        return logits, preds, accuracy.item()

    def predict(self, inputs, gamma=1):
        """ Run a step of forward and backward model update. """
        if self.opt['cuda']:
            inputs = inputs.cuda()

        self.model.eval()

        logits = self.model(inputs) * gamma

        logits = torch.softmax(logits, dim=-1).detach()

        return logits

    # save the model
    def save(self, filename):
        params = {
                'model': self.model.state_dict(), # model parameters
                'config': self.opt, # options
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    # load the model
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']


        
