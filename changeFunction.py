import torch
import math

def change_function(x,m,b):    
    x=(m*x+b).float()

    x=torch.sigmoid(x)

    return x

def change_m_function(x,m,b,label,i):
    cache=(m*x+b).float()

    cache=torch.sigmoid(cache)*(1-torch.sigmoid(cache))

    cache*=x*((-1)**(label==i).float())*(-1)

    return cache

def change_b_function(x,m,b,label,i):
    cache=(m*x+b).float()

    cache=torch.sigmoid(cache)*(1-torch.sigmoid(cache))

    cache=cache*((-1)**(label==i).float())*(-1)

    return cache

    
