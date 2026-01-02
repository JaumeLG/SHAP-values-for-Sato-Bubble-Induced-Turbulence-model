import os
import re
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from time import time
import gc
import shap


print('reading folder')

lista=glob.glob('./data/*')
timesIn=[]
for i in range(len(lista)):
    timesIn.append(lista[i].split("_")[-1].split(".np")[0])

del lista

timesIn = sorted(timesIn, key=float)

inputs=np.empty((len(timesIn)-1,6,60,167),dtype=np.float32)
outputs=np.empty((len(timesIn)-1,1,60,167),dtype=np.float32)

for i in range(len(timesIn)):
    temp=np.load('./data/data_t_'+timesIn[i]+'.npz')
    if i==0:
        inputs[i]=np.array(temp['data'])
    elif i==(len(timesIn)-1):
        outputs[i-1]=np.array(temp['data'][0])
    else:
        inputs[i]=np.array(temp['data'])
        outputs[i-1]=np.array(temp['data'][0])
        
del timesIn    

inputs=inputs[6000:]
outputs=outputs[6000:]


def scale(array):
    dims=array.shape
    scaler=np.zeros((2,dims[1],dims[2]),dtype=np.float32)
    scaledM=np.zeros((array.shape))
    
    mean=np.mean(array,axis=0)
    std=np.std(array,axis=0)
    
    scaledM=(array-mean)/std
    scaler[0]=mean
    scaler[1]=std
    
    return scaledM,scaler
    
scalerList=[]

for i in range(6):
    inputs[:,i,:,:],tempScaler=scale(inputs[:,i,:,:])
    scalerList.append(tempScaler)
    
outputs[:,0,:,:],tempScaler=scale(outputs[:,0,:,:])

"""if resume==0:
    print('Saving scaler')
    np.save('./results2/scalerList.npy', np.array(scalerList,dtype=object),allow_pickle=True)"""

del scalerList
del tempScaler


print('Creating tensor dataset to train')
X=torch.from_numpy(inputs).type(torch.float32)
Y=torch.from_numpy(outputs).type(torch.float32)

tensorDataset=TensorDataset(X,Y)
del X
del Y
del inputs
del outputs
trainData, testData = random_split(tensorDataset,[0.8,0.2],generator=torch.Generator().manual_seed(37))
train_DL = DataLoader(trainData,batch_size=64)#pin_memory=True,num_workers=1
test_DL = DataLoader(testData,batch_size=32)

print('Defining model')

device=torch.device('cuda')

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnv1=nn.Conv2d(6,64,3,1,padding=1)
        self.cnv2=nn.Conv2d(64,128,3,1,padding=1)
        self.cnv3=nn.Conv2d(128,256,3,1,padding=1)
        self.cnv4=nn.Conv2d(256,512,3,1,padding=1)
        self.cnv5=nn.Conv2d(512,1024,3,1,padding=1)
        self.cnv6=nn.ConvTranspose2d(1024,512,3,1,padding=1)
        self.cnv7=nn.ConvTranspose2d(512,256,3,1,padding=1)
        self.cnv8=nn.ConvTranspose2d(256,128,3,1,padding=1)
        self.cnv9=nn.ConvTranspose2d(128,64,3,1,padding=1)
        self.cnv10=nn.ConvTranspose2d(64,32,3,1,padding=1)
        self.cnv11=nn.ConvTranspose2d(32,1,3,1,padding=1)
        self.cnv12=nn.ConvTranspose2d(1,1,3,1,padding=1)

    def forward(self,x):
        x=F.tanh(self.cnv1(x))
        x=F.tanh(self.cnv2(x))
        x=F.tanh(self.cnv3(x))
        x=F.tanh(self.cnv4(x))
        x=F.tanh(self.cnv5(x))
        x=F.tanh(self.cnv6(x))
        x=F.tanh(self.cnv7(x))
        x=F.tanh(self.cnv8(x))
        x=F.tanh(self.cnv9(x))
        x=F.tanh(self.cnv10(x))
        x=F.tanh(self.cnv11(x))
        x=self.cnv12(x)

        return x
        
class CNN_loss(nn.Module):
    def __init__(self,y):
        super().__init__()

        self.cnv1=nn.Conv2d(6,64,3,1,padding=1)
        self.cnv2=nn.Conv2d(64,128,3,1,padding=1)
        self.cnv3=nn.Conv2d(128,256,3,1,padding=1)
        self.cnv4=nn.Conv2d(256,512,3,1,padding=1)
        self.cnv5=nn.Conv2d(512,1024,3,1,padding=1)
        self.cnv6=nn.ConvTranspose2d(1024,512,3,1,padding=1)
        self.cnv7=nn.ConvTranspose2d(512,256,3,1,padding=1)
        self.cnv8=nn.ConvTranspose2d(256,128,3,1,padding=1)
        self.cnv9=nn.ConvTranspose2d(128,64,3,1,padding=1)
        self.cnv10=nn.ConvTranspose2d(64,32,3,1,padding=1)
        self.cnv11=nn.ConvTranspose2d(32,1,3,1,padding=1)
        self.cnv12=nn.ConvTranspose2d(1,1,3,1,padding=1)
        
        self.truth=y

    def forward(self,x):
        x=F.tanh(self.cnv1(x))
        x=F.tanh(self.cnv2(x))
        x=F.tanh(self.cnv3(x))
        x=F.tanh(self.cnv4(x))
        x=F.tanh(self.cnv5(x))
        x=F.tanh(self.cnv6(x))
        x=F.tanh(self.cnv7(x))
        x=F.tanh(self.cnv8(x))
        x=F.tanh(self.cnv9(x))
        x=F.tanh(self.cnv10(x))
        x=F.tanh(self.cnv11(x))
        out=self.cnv12(x)
        
        return torch.mean((out-self.truth)**2,dim=(1,2,3)).reshape(-1,1)     
        
        
Ir_np=np.zeros((1,6,60,167))
Ir=torch.from_numpy(Ir_np).type(torch.float32)   

weights=torch.load('./results/model_2DCNN_Sato',weights_only=True)


def mseLoss(weights,testData,Ir):
    inputT=testData[0]
    shp = inputT.shape
    inputT=inputT.reshape(1,shp[0],shp[1],shp[2])
    inputT=inputT.to(device)
    truthT1=testData[1]
    truthT1=truthT1.reshape(1,1,shp[1],shp[2])
    truthT1=truthT1.to(device)
        
    myModel=CNN_loss(truthT1)
    myModel.load_state_dict(weights)
    myModel.to(device)
    
    
    explainer=shap.GradientExplainer(myModel,Ir)
    shapValues=explainer.shap_values(inputT)
    
    del myModel
    del inputT
    del truthT1
    
    return shapValues
    
Ir=Ir.to(device)
start=time()

for i in range(len(testData)):
    print(f'Iteration: {i}')
    if i==0:
        shapVal=mseLoss(weights,testData[i],Ir)
    else:
        shapVal=np.concatenate((shapVal,mseLoss(weights,testData[i],Ir)),axis=0)
        
    if (i%100==0):
        print('100 iteraciones: ',round(time()-start,2))
        start=time()


np.savez('./shap/shapVal.npz',shapVal)
   