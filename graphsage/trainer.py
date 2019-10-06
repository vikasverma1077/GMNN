import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from graphsage import torch_utils

class Trainer(object):
    """ A wrapper class for the training and evaluation of models. """
    def __init__(self, opt, model):
        # options
        self.opt = opt
        # model
        self.model = model
        # loss function
        self.criterion = nn.BCELoss()
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

    # train the model with a batch
    def update_sigmoid(self, inputs, target, idx):
        """ Run a step of forward and backward model update. """
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(inputs)

        sig = F.sigmoid(logits)
        loss = torch.mean(target * torch.log(sig) + (1 - target) * torch.log(1-sig))
        loss = -loss
        
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict_noisy(self, inputs):
        if self.opt['cuda']:
            inputs = inputs.cuda()

        #self.model.eval()

        logits = self.model(inputs) / self.opt['tau']

        logits = torch.sigmoid(logits).detach()

        return logits

    # train the model with a batch
    def update_gs(self, inputs, target):
        """ Run a step of forward and backward model update. """
        if self.opt['cuda']:
            target = target.cuda()

        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(inputs)
        logits = torch.sigmoid(logits)
        loss = self.criterion(logits, target)
        
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def loss_gs(self, inputs, target):
        """ Run a step of forward and backward model update. """
        if self.opt['cuda']:
            target = target.cuda()

        self.model.train()

        logits = self.model(inputs)
        logits = torch.sigmoid(logits)
        loss = self.criterion(logits, target)
        
        return loss

    def update_soft_aux(self, idx, idx_unlabeled, target, target_unlabeled, mixup_layer):
        """uses the auxiliary loss as well, which does not use the adjacency information"""
        if self.opt['cuda']:
            target = target.cuda()
            idx = idx.cuda()

        self.model.train()
        self.optimizer.zero_grad()

        logits, target_a, target_b, lam = self.model.forward_aux(target=target, train_idx= idx, mixup_input=False, mixup_hidden = True, mixup_alpha = self.opt['mixup_alpha'],layer_mix=mixup_layer)
        mixed_target = lam*target_a + (1-lam)*target_b
        loss = self.criterion(torch.sigmoid(logits), mixed_target)

        logits, target_a, target_b, lam = self.model.forward_aux(target=target_unlabeled, train_idx= idx_unlabeled, mixup_input=False, mixup_hidden = True, mixup_alpha = self.opt['mixup_alpha'],layer_mix= mixup_layer)
        mixed_target = lam*target_a + (1-lam)*target_b
        loss_usup = self.criterion(torch.sigmoid(logits), mixed_target)
        
        return loss, loss_usup
    
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

        return loss.item(), preds, accuracy.item()

    def predict(self, inputs, gamma=1):
        """ Run a step of forward and backward model update. """
        if self.opt['cuda']:
            inputs = inputs.cuda()

        self.model.eval()

        logits = self.model(inputs) * gamma

        logits = torch.softmax(logits, dim=-1).detach()

        return logits

    def change_lr(self, new_lr):

        torch_utils.change_lr(self.optimizer, new_lr)

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


        
