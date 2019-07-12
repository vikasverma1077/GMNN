import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Optimizer

bce_loss = nn.BCELoss().cuda()
softmax = nn.Softmax(dim=1).cuda()
class_criterion = nn.CrossEntropyLoss().cuda()
def mixup_criterion(y_a, y_b, lam):
        return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_optimizer(name, parameters, lr, weight_decay=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))

def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

class Trainer(object):
    def __init__(self, opt, model):
        self.opt = opt
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.criterion.cuda()
        self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])

    def reset(self):
        self.model.reset()
        self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])

    def update(self, inputs, target, idx):
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

    def update_soft(self, inputs, target, idx):
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
    
    def update_soft_mlp(self, inputs, target, idx):
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.train()
        self.optimizer.zero_grad()
        temp = inputs[idx]
        target_temp = target[idx]
       
        rand_idx = torch.randint(0, idx.shape[0]-1, (32,))
        #import pdb; pdb.set_trace()
        logits, mixed_target = self.model(temp[rand_idx], target=target_temp[rand_idx], mixup_input = False, mixup_hidden= True, mixup_alpha=1.0,layer_mix=4)
        #logits = torch.log_softmax(logits, dim=-1)
        
        #loss = -torch.mean(torch.sum(t * logits, dim=-1))
    
        #loss_func = mixup_criterion(y_a, y_b, lam)
        #class_loss = loss_func(class_criterion, output_mixed_l)
        loss = bce_loss(softmax(logits), mixed_target)

        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def evaluate(self, inputs, target, idx):
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

    def predict(self, inputs, tau=1):
        if self.opt['cuda']:
            inputs = inputs.cuda()

        self.model.eval()

        logits = self.model(inputs) / tau

        logits = torch.softmax(logits, dim=-1).detach()

        return logits

    def save(self, filename):
        params = {
                'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict()
                }
        try:
            torch.save(params, filename)
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optim'])
