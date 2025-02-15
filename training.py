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

model_file="AIdata2.txt"

image_path="train-images.idx3-ubyte"
label_path="train-labels.idx1-ubyte"

images=reading.load_images(image_path)/255.0
labels=reading.load_labels(label_path)
images = torch.tensor(images, dtype=torch.float64).to(torch.device('cuda'))
labels = torch.tensor(labels, dtype=torch.long).to(torch.device('cuda'))
mask=(labels==0) | (labels==1)
images=images[mask]
labels=labels[mask]

images=(images-images.mean())/images.std()

image_size=(images.shape[1],images.shape[2])

models=[]

train_num=10000

learn_rate=200

images=images[:train_num]
labels=labels[:train_num]

print(image_size)

def direc(a):
    if a>0:
        return 1
    return -1

class kernel:
    def __init__(self,size=5):
        self.size=size
        self.data=torch.rand(self.size,self.size, dtype=torch.float64)*2-1
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
            self.juanjihes.append(kernel())

        self.model=[]

        for i in range(576): #576 is the number of input nodes
            self.model.append([])
            for j in range(50): #50 is the number of hidden nodes
                self.model[i].append([])
                for k in range(2):
                    self.model[i][j].append(random.uniform(-2,2))
        
        self.model=torch.tensor(self.model, dtype=torch.float64).to(torch.device('cuda'))

        self.model_1=[]

        for i in range(self.model.size()[1]):
            self.model_1.append([])
            for j in range(1): #1 is the number of output nodes
                self.model_1[i].append([])
                for k in range(2):
                    self.model_1[i][j].append(random.uniform(-2,2))
        
        self.model_1=torch.tensor(self.model_1, dtype=torch.float64).to(torch.device('cuda'))

        self.ans=[]
        for i in range(self.model_1.size()[1]):
            self.ans.append(0)

        self.ans=torch.tensor(self.ans, dtype=torch.float64).to(torch.device('cuda'))

        self.loss=0

        self.changed=(0,0,0)
        self.changed_1=(0,0,0)
        self.changed_kernel=(0,0,0)
    
    def apply(self,image):
        image=image.clone().unsqueeze(0).unsqueeze(0)

        for i in range(len(self.juanjihes)):
            image=self.juanjihes[i].apply(image)

        #5*5
        image=image[0][0]
        image=torch.flatten(image)

        #print(image.size())

        cache=torch.empty((self.model.size()[1]),dtype=torch.float64).to("cuda")

        for i in range(self.model.size()[1]):
            cache[i]=float(change_function(image,self.model[:,i,0],self.model[:,i,1]).sum())

        for i in range(self.model_1.size()[1]):
            self.ans[i]=float(change_function(cache,self.model_1[:,i,0],self.model_1[:,i,1]).sum())

        self.ans=torch.sigmoid(self.ans)
        
    def change(self,factor=1):
        self.model+=self.changed*factor
        self.model_1+=self.changed_1*factor
        self.juanjihes[0].data[0][0]+=self.changed_kernel*factor

    def deri_change(self,images=images,labels=labels):
        global learn_rate
        image=images.clone().unsqueeze(1)

        for i in range(len(self.juanjihes)):
            image=self.juanjihes[i].apply(image)
        
        image=image.squeeze()
        image=torch.flatten(image,1)

        print(images.size(),"1111")

        derivatives=torch.empty((self.model.size()[0], self.model.size()[1], self.model.size()[2], image.size()[0]),dtype=torch.float64).to("cuda")
        derivatives_1=torch.empty((self.model_1.size()[0],self.model_1.size()[1],self.model_1.size()[2],image.size()[0]),dtype=torch.float64).to("cuda")

        cache=torch.empty((self.model.size()[1],image.size()[0]),dtype=torch.float64).to("cuda")

        for i in range(self.model.size()[1]):
            for j in range(image.size()[0]):
                cache[i][j]=float(change_function(image[j],self.model[:,i,0],self.model[:,i,1]).mean())

        #由于最终结果有三层嵌套：changefuncion，sigmoid和-log，因此应用chain rule三次
        for i in range(self.model_1.size()[0]):
            for j in range(self.model_1.size()[1]):
                derivatives_1[i][j][0]=change_m_function(cache[i],self.model_1[i][j][0],self.model_1[i][j][1])
                cachecache=torch.sigmoid(change_function(cache[i],self.model_1[i][j][0],self.model_1[i][j][1]))
                derivatives_1[i][j][0]*=cachecache*(1-cachecache)
                derivatives_1[i][j][0]*=-1/(1*labels+((-1)**labels)*cachecache+0.0000001)*((-1)**labels)

        for i in range(self.model_1.size()[0]):
            for j in range(self.model_1.size()[1]):
                derivatives_1[i][j][1]=change_b_function(cache[i],self.model_1[i][j][0],self.model_1[i][j][1])
                cachecache=torch.sigmoid(change_function(cache[i],self.model_1[i][j][0],self.model_1[i][j][1]))
                derivatives_1[i][j][1]*=cachecache*(1-cachecache)
                derivatives_1[i][j][1]*=-1/(1*labels+((-1)**labels)*cachecache+0.0000001)*((-1)**labels)


        for i in range(self.model.size()[0]):
            for j in range(self.model.size()[1]):
                derivatives[i][j][0]=change_m_function(image[:,i],self.model[i][j][0],self.model[i][j][1])

        for i in range(self.model.size()[0]):
            for j in range(self.model.size()[1]):
                derivatives[i][j][1]=change_b_function(image[:,i],self.model[i][j][0],self.model[i][j][1])


        #cache_1最后一层关于倒数第二层输入数据的导数
        cache_1=torch.empty((self.model_1.size()[0],self.model_1.size()[1],image.size()[0]),dtype=torch.float64).to("cuda")

        for i in range(self.model_1.size()[0]):
            for j in range(self.model_1.size()[1]):
                cache_1[i][j]=change_m_function(self.model_1[i][j][0],cache[i],self.model_1[i][j][1])
                cachecache=torch.sigmoid(change_function(cache[i],self.model_1[i][j][0],self.model_1[i][j][1]))
                cache_1[i][j]*=cachecache*(1-cachecache)
                cache_1[i][j]*=-1/(1*labels+((-1)**labels)*cachecache+0.0000001)*((-1)**labels)

        cache_1=cache_1.mean(dim=-1)
        cache_1=cache_1.mean(dim=-1)

        #倒数第二层的倒数乘以上一层的梯度

        for i in range(self.model.size()[0]):
            for j in range(self.model.size()[1]):
                for k in range(self.model.size()[2]):
                    derivatives[i][j][k]*=cache_1[j]
                    #cache应是关于x的导数，而非关于m和b的倒数。只需要调用change_m_function但是把x和m的位置换一下即可

        #计算卷积核梯度

        cache_2=torch.empty((self.model.size()[0],self.model.size()[1],image.size()[0]),dtype=torch.float64).to("cuda")
        for i in range(self.model.size()[0]):
            for j in range(self.model.size()[1]):
                cache_2[i][j]=change_m_function(self.model[i][j][0],image[:,i],self.model[i][j][1])

        cache_2=cache_2.mean(dim=-1)
        for i in range(self.model.size()[0]):
            cache_2[i]*=cache_1

        cache_2=cache_2.mean(dim=-1)

        #cache_2现为第一层关于卷积操作结果的导数
        #计算卷积核每个参数的导数
        deri_kernel=torch.empty((5,5,images.size()[0]),dtype=torch.float64).to("cuda")
        for i in range(5):
            for j in range(5):
                for k in range(24):
                    for l in range(24):
                        deri_kernel[i][j]+=cache_2[k*24+l]*images[:,k+i,l+j]
        
        #求数据点关于所有图平均的导数
        
        derivatives_1=torch.mean(derivatives_1,dim=-1)
        derivatives=torch.mean(derivatives,dim=-1)
        deri_kernel=deri_kernel.mean(dim=-1)

        print(derivatives_1.sum())
        print(derivatives.sum())
        print(deri_kernel.sum())

        derivatives_1=torch.clamp(derivatives_1,max=10,min=-10)
        derivatives=torch.clamp(derivatives,max=10,min=-10)
        deri_kernel=torch.clamp(deri_kernel,max=10,min=-10)

        print(derivatives_1.sum())
        print(derivatives.sum())
        print(deri_kernel.sum())

        self.changed_1=derivatives_1
        self.changed=derivatives
        self.changed_kernel=deri_kernel
        
        self.change(learn_rate)

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

def best_threshold(scores, labels):
    
    # 生成候选阈值，包括所有评分点和两端额外的值
    thresholds = torch.unique(scores)
    thresholds = torch.cat([torch.tensor([0.0]).to("cuda"), thresholds, torch.tensor([1.0]).to("cuda")])
    
    best_thresh = 0.0
    best_acc = 0.0
    
    for thresh in thresholds:
        predictions = (scores <= thresh).int()
        acc = (predictions == labels).float().mean().item()
        
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh.item()
    
    return best_thresh, best_acc

def apply(k,train_num=train_num):
    scores=[]
    thres=0.5606555236491332
    
    k.loss=0

    cnt=0
    
    for j in range(train_num):
        k.apply(images[j])
        print(float(k.ans[0]),int(labels[j]))
        scores.append(k.ans[0].item())
        if labels[j]==1:
            k.loss-=math.log(float(1-k.ans[0]+0.0001))#这里搞反了，所以这个模型，答案越接近0，越有可能是1
        else:
            k.loss-=math.log(float(k.ans[0]+0.0001))
        if k.ans[0]>thres and labels[j]==0:
            cnt+=1
        if k.ans[0]<=thres and labels[j]==1:
            cnt+=1
    print("accu: ",cnt/train_num)
    
    scores=torch.tensor(scores,dtype=torch.float64).to("cuda")
    print("best thres, best acc:", best_threshold(scores,labels))

    if cnt/train_num>0.8:
        k.write(str(k.loss)+'.txt')
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
    
    models[1].change(-1*learn_rate)
    models[1].change(learn_rate*direc(models[0].loss-models[1].loss))
    models[0].clone(models[1])
    
    apply(models[0])

    print(models[0].loss,cache.loss)
    if models[0].loss>=cache.loss+0.1:
        models[0].clone(cache)
        learn_rate*=0.7
    else:
        if models[0].loss>=cache.loss:
            learn_rate*=0.7
        elif random.randint(0,2)==0:
            learn_rate/=0.7
        cache.clone(models[0])
    print(learn_rate)
    print(models[0].loss)
    print()

print(models[0].loss)
print(datetime.datetime.now())

if abs(models[0].loss-initial)<0.0000000001:
    cache=apply(models[0])
    if cache>0.7:
        models[0].write(str(models[0].loss)+'.txt')
    '''
    if os.path.exists(model_file):
        os.remove(model_file)
    os.startfile("training.py")'''
else:
    models[0].write()
    os.startfile("training.py")
