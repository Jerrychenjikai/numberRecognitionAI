import torch
import math

def change_function(x,m,b):
    decrease=1
    
    x=(m*x+b).float()

    x=torch.sigmoid(x)

    #x=x/decrease

    return x

def change_m_function(x,m,b,label,i):
    decrease=1

    cache=(m*x+b).float()

    cache=torch.sigmoid(cache)*(1-torch.sigmoid(cache))

    cache*=x/decrease*((-1)**(label==i).float())

    return cache

def change_b_function(x,m,b,label,i):
    decrease=1

    cache=(m*x+b).float()

    cache=torch.sigmoid(cache)*(1-torch.sigmoid(cache))

    cache=cache/decrease*((-1)**(label==i).float())

    return cache

    
