import torch
import math

factor=0.02

def change_function(x,m,b):
    global factor
    x=(m*x+b).float()

    x=torch.sigmoid(x)

    return x*factor

def change_m_function(x,m,b):
    global factor
    cache=(m*x+b).float()

    cache=torch.sigmoid(cache)*(1-torch.sigmoid(cache))

    cache*=x

    return cache*factor

def change_b_function(x,m,b):
    global factor
    cache=(m*x+b).float()

    cache=torch.sigmoid(cache)*(1-torch.sigmoid(cache))

    cache=cache

    return cache*factor

    
