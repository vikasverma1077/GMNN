import random
import time
import torch

def sparse_dropout(x, drop_rate, training): 

    #print('training', training)

    x_drop = x*1.0

    if training: 
        #print('dropout train mode')
        vals = x_drop._values()
        bern = torch.bernoulli(torch.ones_like(vals) * (1.0 - drop_rate))
        vals *= bern / (1.0 - drop_rate)
    else:
        #print('dropout test mode')
        x_drop = x_drop

    return x_drop


