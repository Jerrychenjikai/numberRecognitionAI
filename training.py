import reading
import cv2
import math
import os
import json
import random
import copy
import torch
import datetime
from changeFunction import *
import quicksort

model_file="AIdata2.txt"

image_path="train-images.idx3-ubyte"
label_path="train-labels.idx1-ubyte"

images=reading.load_images(image_path)/255.0
labels=reading.load_labels(label_path)
images = torch.tensor(images, dtype=torch.float32).to(torch.device('cuda'))
labels = torch.tensor(labels, dtype=torch.long).to(torch.device('cuda'))

image_size=(images.shape[1],images.shape[2])

models=[]

train_num=500

learn_rate=10000

images=images[:train_num]
labels=labels[:train_num]

print(image_size)

class juanjihe:
    def __init__(self,size=3):
        self.size=size
        self.data=torch.rand(self.size,self.size)*4-2
        self.data=self.data.to('cuda').unsqueeze(0).unsqueeze(0)

    def apply(self,image):
        image=image.clone()

        return torch.nn.functional.conv2d(image,self.data)

    def write(self,model_file=model_file):
        with open(model_file,'a') as fo:
            for i in range(self.size):
                for j in range(self.size):
                    fo.write(str(float(self.data[0][0][i][j]))+'\n')
    def read(self,file):
        for i in range(self.size):
            for j in range(self.size):
                self.data[0][0][i][j]=float(file[i*self.size+j])
            
    def clone(self,sample):
        self.data=sample.data.clone()

class model:
    def __init__(self):
        self.juanjihes=[]
        for i in range(1):
            self.juanjihes.append(juanjihe())

        self.model=[]

        for i in range(10):
            self.model.append([])
            for j in range(169):
                self.model[i].append([])
                for k in range(2):
                    self.model[i][j].append(random.uniform(-2,2))
        
        self.model=torch.tensor(self.model, dtype=torch.float32).to(torch.device('cuda'))

        self.ans=[]
        for i in range(self.model.size()[0]):
            self.ans.append(0)

        self.ans=torch.tensor(self.ans, dtype=torch.float32).to(torch.device('cuda'))

        self.loss=0

        self.changed=(0,0,0)
    
    def apply(self,image):
        image=image.clone().unsqueeze(0).unsqueeze(0)

        for i in range(1):
            image=self.juanjihes[i].apply(image)
            image=torch.nn.functional.max_pool2d(image,2,2,0)

        #5*5
        image=image[0][0]
        image=torch.flatten(image)

        for i in range(self.model.size()[0]):
            self.ans[i]=float(change_function(image,self.model[i][:,0],self.model[i][:,1]).sum())

        self.ans=torch.nn.functional.softmax(self.ans,dim=-1)
        
    def change(self):
        self.model[self.changed[0]][self.changed[1]][self.changed[2]]+=self.changed[3]

    def deri_change(self,images=images,labels=labels):
        image=images.clone().unsqueeze(1)

        for i in range(1):
            image=self.juanjihes[i].apply(image)
            image=torch.nn.functional.max_pool2d(image,2,2,0)
        
        image=image.squeeze()
        image=torch.flatten(image,1)

        derivatives=torch.empty((self.model.size()[2],self.model.size()[0],self.model.size()[1]),dtype=torch.float32)

        for i in range(self.model.size()[0]):
            for j in range(self.model.size()[1]):
                derivatives[0][i][j]=change_m_function(image[:,j],self.model[i][j][0],self.model[i][j][1],labels[:],i).sum()

        for i in range(self.model.size()[0]):
            for j in range(self.model.size()[1]):
                derivatives[1][i][j]=change_b_function(image[:,j],self.model[i][j][0],self.model[i][j][1],labels[:],i).sum()

        max_index=derivatives.argmax()

        self.changed=(int(int(max_index%(derivatives.shape[2]*derivatives.shape[1]))/derivatives.shape[2]),
                      int(max_index%derivatives.shape[2]),
                      int(max_index/derivatives.shape[2]/derivatives.shape[1]),
                      0.1)

        self.change()

    def clone(self,sample):
        for i in range(len(self.juanjihes)):
            self.juanjihes[i].clone(sample.juanjihes[i])

        self.loss=sample.loss

        self.model=sample.model.clone()
        self.ans=sample.ans.clone()

        self.changed=sample.changed

    def write(self,model_file=model_file):
        with open(model_file,'w') as fo:
            fo.close()
        
        for i in self.juanjihes:
            i.write(model_file)

        with open(model_file,'a') as fo:
            for i in range(self.model.size()[0]):
                for j in range(self.model.size()[1]):
                    for k in range(self.model.size()[2]):
                        fo.write(str(float(self.model[i][j][k]))+'\n')
            fo.close()
    def read(self,model_file=model_file):
        with open(model_file,'r') as fo:
            file=fo.readlines()
            fo.close()

        for i in self.juanjihes:
            i.read(file)
            file=file[i.size*i.size:]

        for i in range(self.model.size()[0]):
            for j in range(self.model.size()[1]):
                for k in range(self.model.size()[2]):
                    self.model[i][j][k]=float(file[i*self.model.size()[2]*self.model.size()[1]+j*self.model.size()[2]+k])

def apply(k,train_num=train_num):
    k.loss=0

    cnt=0
    
    for j in range(train_num):
        k.apply(images[j])
        k.loss+=(float(1-k.ans[labels[j]])*2)**(2)
        if max(k.ans)==k.ans[labels[j]]:
            k.loss-=0.5
            cnt+=1
    print("accu: ",cnt/train_num)
    return cnt/train_num

print(datetime.datetime.now())
print(model_file)

for i in range(2):
    models.append(model())

if os.path.exists(model_file):
    models[0].read()

apply(models[0])
cache=model()
cache.clone(models[0])

for i in range(10):
    models[1].clone(models[0])
    models[1].deri_change()
    apply(models[1])

    models[1].model[models[1].changed[0]][models[1].changed[1]][models[1].changed[2]]-=0.1
    models[1].model[models[1].changed[0]][models[1].changed[1]][models[1].changed[2]]+=(models[0].loss-models[1].loss)*learn_rate
    models[0].clone(models[1])
    #'''
    apply(models[0])
    if models[0].loss>=cache.loss:
        models[0].clone(cache)
        learn_rate/=10
    else:
        cache.clone(models[0])#'''
        if random.randint(0,3)==0:
            learn_rate*=10
    print(learn_rate)
    print(models[0].loss)

print(models[0].loss)
print(datetime.datetime.now())

if learn_rate<0.1:
    cache=apply(models[0])
    if cache>0.2:
        models[0].write(str(models[0].loss)+'.txt')
    if os.path.exists(model_file):
        os.remove(model_file)
    os.startfile("training.py")
else:
    models[0].write()
    os.startfile("training.py")
