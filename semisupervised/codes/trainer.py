import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from gcn import GCN
import torch_utils


bce_loss = nn.BCELoss().cuda()
softmax = nn.Softmax(dim=1).cuda()
class_criterion = nn.CrossEntropyLoss().cuda()
def mixup_criterion(y_a, y_b, lam):
            return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Trainer(object):
    """ A wrapper class for the training and evaluation of models. """
    def __init__(self, opt, model, ema= False):
        # options
        self.ema = ema
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
        if  self.ema == False:
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
        
        #loss.backward()
        #self.optimizer.step()
        return loss
    
    
    def update_soft_aux(self, inputs, target,target_discrete, idx, idx_unlabeled, opt, mixup_layer):
        """uses the auxiliary loss as well, which does not use the adjacency information"""
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()
            idx_unlabeled = idx_unlabeled.cuda()

        self.model.train()
        self.optimizer.zero_grad()

        #import pdb ;pdb.set_trace()
        mixup = True
        if mixup == True:
            # get the supervised mixup loss #
            logits, target_a, target_b, lam = self.model.forward_aux(inputs, target=target, train_idx= idx, mixup_input=False, mixup_hidden = True, mixup_alpha = opt['mixup_alpha'],layer_mix=mixup_layer)
            #import pdb; pdb.set_trace()
            mixed_target = lam*target_a + (1-lam)*target_b
            #logits = torch.log_softmax(logits, dim=-1)
            #loss_aux = -(torch.mean(lam*torch.sum(target_a * logits[idx], dim=-1, keepdim= True))+ torch.mean((1-lam)*torch.sum(target_b * logits[idx], dim=-1, keepdim =True)))
            loss = bce_loss(softmax(logits[idx]), mixed_target)

            # get the unsupervised mixup loss #
            logits, target_a, target_b, lam = self.model.forward_aux(inputs, target=target, train_idx= idx_unlabeled, mixup_input=False, mixup_hidden = True, mixup_alpha = opt['mixup_alpha'],layer_mix= mixup_layer)
            mixed_target = lam*target_a + (1-lam)*target_b
            loss_usup = bce_loss(softmax(logits[idx_unlabeled]), mixed_target)
        else:
            logits = self.model.forward_aux(inputs, target=None, train_idx= idx, mixup_input= False, mixup_hidden = False, mixup_alpha = 0.0,layer_mix=None)
            logits = torch.log_softmax(logits, dim=-1)
            loss = -torch.mean(torch.sum(target[idx] * logits[idx], dim=-1))


            logits = self.model.forward_aux(inputs, target=None, train_idx= idx_unlabeled, mixup_input= False, mixup_hidden = False, mixup_alpha = 0.0,layer_mix=None)
            logits = torch.log_softmax(logits, dim=-1)
            loss_usup = -torch.mean(torch.sum(target[idx_unlabeled] * logits[idx_unlabeled], dim=-1))

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

        return logits, preds, accuracy.item()

    def predict(self, inputs, gamma=1):
        """ Run a step of forward and backward model update. """
        if self.opt['cuda']:
            inputs = inputs.cuda()

        self.model.eval()

        logits = self.model(inputs) * gamma

        logits = torch.softmax(logits, dim=-1).detach()

        return logits
    
    
    def predict_noisy(self, inputs, tau=1):
        if self.opt['cuda']:
            inputs = inputs.cuda()
        inputs = F.dropout(inputs, 0.5, training=True)
        #self.model.eval()
        logits = self.model(inputs) / tau

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


        
