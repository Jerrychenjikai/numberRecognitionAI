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

images=(images-images.mean())/images.std()

image_size=(images.shape[1],images.shape[2])

models=[]

train_num=1000

learn_rate=10

images=images[:train_num]
labels=labels[:train_num]

print(image_size)

class juanjihe:
    def __init__(self,size=5):
        self.size=size
        self.data=torch.rand(self.size,self.size)*2-1
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

        for i in range(576):
            self.model.append([])
            for j in range(20):
                self.model[i].append([])
                for k in range(2):
                    self.model[i][j].append(random.uniform(-1,1))
        
        self.model=torch.tensor(self.model, dtype=torch.float32).to(torch.device('cuda'))

        self.model_1=[]

        for i in range(self.model.size()[1]):
            self.model_1.append([])
            for j in range(10):
                self.model_1[i].append([])
                for k in range(2):
                    self.model_1[i][j].append(random.uniform(-1,1))
        
        self.model_1=torch.tensor(self.model_1, dtype=torch.float32).to(torch.device('cuda'))

        self.ans=[]
        for i in range(self.model_1.size()[1]):
            self.ans.append(0)

        self.ans=torch.tensor(self.ans, dtype=torch.float32).to(torch.device('cuda'))

        self.loss=0

        self.changed=(0,0,0)

        self.changed_1=(0,0,0)
    
    def apply(self,image):
        image=image.clone().unsqueeze(0).unsqueeze(0)

        for i in range(len(self.juanjihes)):
            image=self.juanjihes[i].apply(image)

        #5*5
        image=image[0][0]
        image=torch.flatten(image)

        #print(image.size())

        cache=torch.empty((self.model.size()[1]),dtype=torch.float32).to("cuda")

        for i in range(self.model.size()[1]):
            cache[i]=float(change_function(image,self.model[:,i,0],self.model[:,i,1]).sum())

        for i in range(self.model_1.size()[1]):
            self.ans[i]=float(change_function(cache,self.model_1[:,i,0],self.model_1[:,i,1]).sum())

        self.ans=torch.nn.functional.softmax(self.ans,dim=-1)
        
    def change(self):
        self.model+=self.changed
        self.model_1+=self.changed_1

    def deri_change(self,images=images,labels=labels):
        image=images.clone().unsqueeze(1)

        for i in range(len(self.juanjihes)):
            image=self.juanjihes[i].apply(image)
        
        image=image.squeeze()
        image=torch.flatten(image,1)

        print(image.size(),"1111")

        derivatives=torch.empty((self.model.size()[0],self.model.size()[1],self.model.size()[2],image.size()[0]),dtype=torch.float32).to("cuda")
        derivatives_1=torch.empty((self.model_1.size()[0],self.model_1.size()[1],self.model_1.size()[2],image.size()[0]),dtype=torch.float32).to("cuda")

        cache=torch.empty((self.model.size()[1],image.size()[0]),dtype=torch.float32).to("cuda")

        for i in range(self.ans.size()[0]):
            for j in range(image.size()[0]):
                cache[i][j]=float(change_function(image[j],self.model[:,i,0],self.model[:,i,1]).sum())

        for i in range(self.model_1.size()[0]):
            for j in range(self.model_1.size()[1]):
                derivatives_1[i][j][0]=change_m_function(cache[i],self.model_1[i][j][0],self.model_1[i][j][1],labels[:],j)
                derivatives_1[i][j][0]*=-1/(change_function(cache[i],self.model_1[i][j][0],self.model_1[i][j][1])+0.0000001)

        for i in range(self.model_1.size()[0]):
            for j in range(self.model_1.size()[1]):
                derivatives_1[i][j][1]=change_b_function(cache[i],self.model_1[i][j][0],self.model_1[i][j][1],labels[:],j)
                derivatives_1[i][j][1]*=-1/(change_function(cache[i],self.model_1[i][j][0],self.model_1[i][j][1])+0.0000001)


        for i in range(self.model.size()[0]):
            for j in range(self.model.size()[1]):
                derivatives[i][j][0]=change_m_function(image[:,i],self.model[i][j][0],self.model[i][j][1],labels[:],labels[:])

        for i in range(self.model.size()[0]):
            for j in range(self.model.size()[1]):
                derivatives[i][j][1]=change_b_function(image[:,i],self.model[i][j][0],self.model[i][j][1],labels[:],labels[:])

        cache=derivatives_1.sum(dim=1)

        for i in range(self.model.size()[0]):
            for j in range(self.model.size()[1]):
                for k in range(self.model.size()[2]):
                    derivatives[i][j][k]*=cache[j][k]

        print(derivatives_1.sum())
        print(derivatives.sum())
        
        derivatives_1=torch.mean(derivatives_1,dim=-1)
        derivatives=torch.mean(derivatives,dim=-1)

        self.changed_1=derivatives_1
        self.changed=derivatives
        
        self.change()

    def clone(self,sample):
        for i in range(len(self.juanjihes)):
            self.juanjihes[i].clone(sample.juanjihes[i])

        self.loss=sample.loss

        self.model=sample.model.clone()
        self.model_1=sample.model_1.clone()
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

            for i in range(self.model_1.size()[0]):
                for j in range(self.model_1.size()[1]):
                    for k in range(self.model_1.size()[2]):
                        fo.write(str(float(self.model_1[i][j][k]))+'\n')
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
        file=file[self.model.size()[0]*self.model.size()[1]*self.model.size()[2]:]

        for i in range(self.model_1.size()[0]):
            for j in range(self.model_1.size()[1]):
                for k in range(self.model_1.size()[2]):
                    self.model_1[i][j][k]=float(file[i*self.model_1.size()[2]*self.model_1.size()[1]+j*self.model_1.size()[2]+k])

def apply(k,train_num=train_num):
    k.loss=0

    cnt=0
    
    for j in range(train_num):
        k.apply(images[j])
        k.loss-=math.log(float(k.ans[labels[j]]))
        if max(k.ans)==k.ans[labels[j]]:
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

initial=models[0].loss

for i in range(10):
    models[1].clone(models[0])
    models[1].deri_change()
    apply(models[1])
    
    models[1].model-=models[1].changed
    models[1].model_1-=models[1].changed_1
    models[1].model+=models[1].changed*learn_rate*(models[0].loss-models[1].loss)
    models[1].model_1+=models[1].changed_1*learn_rate*(models[0].loss-models[1].loss)
    models[0].clone(models[1])
    
    apply(models[0])

    print(models[0].loss,cache.loss)
    if models[0].loss>=cache.loss+0.1:
        models[0].clone(cache)
        learn_rate*=0.9
    else:
        if models[0].loss>=cache.loss:
            learn_rate*=0.9
        elif random.randint(0,3)==0:
            learn_rate/=0.9
        cache.clone(models[0])
    print(learn_rate)
    print(models[0].loss)
    print()

print(models[0].loss)
print(datetime.datetime.now())

if abs(models[0].loss-initial)<2 or learn_rate<0.01:
    cache=apply(models[0])
    if cache>0.25:
        models[0].write(str(models[0].loss)+'.txt')
    if os.path.exists(model_file):
        os.remove(model_file)
    os.startfile("training.py")
else:
    models[0].write()
    os.startfile("training.py")
