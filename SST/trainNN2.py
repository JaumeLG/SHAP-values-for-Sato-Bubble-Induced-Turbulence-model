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

resume=0

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


#totalInputs=totalInputs[::2]
#totalOutputs=totalOutputs[::2]


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

if resume==0:
    print('Saving scaler')
    np.save('./results/scalerList.npy', np.array(scalerList,dtype=object),allow_pickle=True)

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
        
class blockDir(nn.Module):
    def __init__(self,in_ch,out_ch,krn):
        super().__init__()
        self.conv=nn.Conv2d(in_ch,out_ch,krn,1,padding="same")
        self.btchNorm=nn.BatchNorm2d(out_ch)
        
    def forward(self,x):
        x=self.conv(x)
        #x=self.btchNorm(x)
        return F.tanh(x)#,negative_slope=0.1
    
class blockInv(nn.Module):
    def __init__(self,in_ch,out_ch,krn,pd):
        super().__init__()
        self.conv=nn.ConvTranspose2d(in_ch,out_ch,krn,1,padding=pd)
        self.btchNorm=nn.BatchNorm2d(out_ch)
        
    def forward(self,x):
        x=self.conv(x)
        #x=self.btchNorm(x)
        return F.tanh(x)
        
class UNet(nn.Module):
    def __init__(self,in_sz,out_sz):
        super().__init__()
        
        self.b1=blockDir(in_sz,32,3)
        self.b2=blockDir(32,64,3)
        self.b3=blockDir(64,64,3)
        
        self.b4=blockDir(64,64,3)
        self.b5=blockDir(64,96,3)
        self.b6=blockDir(96,96,3)
        
        self.b7=blockDir(96,96,1)
        self.b8=blockDir(96,128,1)
        self.b9=blockDir(128,256,1)
        self.b10=blockInv(256,128,1,0)
        self.b11=blockInv(128,96,1,0)
        
        self.b12=blockInv(192,96,3,1)
        self.b13=blockInv(96,64,3,1)
        self.b14=blockInv(64,64,3,1)
        
        self.b15=blockInv(128,64,3,1)
        self.b16=blockInv(128,64,3,1)
        self.b17=blockInv(96,32,1,0)
        self.b18=blockInv(32,out_sz,1,0)
        self.out=nn.ConvTranspose2d(out_sz,out_sz,1,1,padding=0)
        
        self.pool=nn.AvgPool2d(2)
#         self.up=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1=nn.ConvTranspose2d(96,96,3,(2,2),padding=1)
        self.up2=nn.ConvTranspose2d(64,64,3,(2,2),padding=1)
        
    def forward(self,x):
        
        x11=self.b1(x)
        x12=self.b2(x11)
        x13=self.b3(x12)
        
        x2=self.pool(x13)
        x2=self.b4(x2)
        x2=self.b5(x2)
        x2=self.b6(x2)
        
        x3=self.pool(x2)
        x3=self.b7(x3)
        x3=self.b8(x3)
        x3=self.b9(x3)
        x3=self.b10(x3)
        x3=self.b11(x3)
        
        x3=self.up1(x3)
        x3=F.pad(x3,(1,1,0,1),"constant",0)
        x4=torch.cat([x2,x3],dim=1)
        x4=self.b12(x4)
        x4=self.b13(x4)
        x4=self.b14(x4)
        
        x4=self.up2(x4)
        x4=F.pad(x4,(1,1,0,1),"constant",0)
        x5=torch.cat([x13,x4],dim=1)
        x5=self.b15(x5)
        x6=torch.cat([x12,x5],dim=1)
        x6=self.b16(x6)
        x7=torch.cat([x11,x6],dim=1)
        x7=self.b17(x7)
        x7=self.b18(x7)
        out=self.out(x7)
        
        return out
        
print('Instantiating model, loss function and optimizer')

if resume==0:

    def init_weights(m):
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.ConvTranspose3d):
            torch.nn.init.xavier_uniform_(m.weight)
        
    #model = UNet(6,1)
    model = CNN()
    model = model.to(device)
    model.apply(init_weights)
    
else:
    weights=torch.load('./results2/model_2DCNN_SST2', weights_only=True)#,map_location=torch.device('cpu')
    model = CNN()
    model.load_state_dict(weights)
    model = model.to(device)

lossF = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,eps=1e-7)#4.69e-4
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.5)

print('Training model')
epochs = 500

if resume==0:
    train_loss = torch.zeros(1).to(device)
    test_loss = torch.zeros(1).to(device)
    
else:
    train_loss=np.array(np.load('./results2/trainHist.npy',allow_pickle=True),dtype=np.float32)
    test_loss=np.array(np.load('./results2/testHist.npy',allow_pickle=True),dtype=np.float32)
        
    train_loss=torch.from_numpy(train_loss).type(torch.float32)
    train_loss=train_loss.to(device)
    #train_loss=train_loss.tolist()
    test_loss=torch.from_numpy(test_loss).type(torch.float32)
    test_loss=test_loss.to(device)
    #test_loss=test_loss.tolist()
    

start=time()
for i in range(epochs):
    avgTrainLoss = 0
    idx = 0
    
    epoch_start=time()
    for k,(x_train,y_train) in enumerate(train_DL):
        optimizer.zero_grad()
        x_train=x_train.to(device)
        y_train=y_train.to(device)
        
        with torch.autocast(device_type="cuda"):
            y_pred = model.forward(x_train)
            batchLoss = lossF(y_pred,y_train)
            
        avgTrainLoss += batchLoss
        
        batchLoss.backward()
        optimizer.step()
    
    temp=avgTrainLoss/(k+1)
    if temp.dim()==0:
        temp=torch.tensor([temp])
        
    temp=temp.to(device)
    if i==0:
        train_loss=torch.tensor(temp)
    else:
        train_loss=torch.cat((train_loss,temp),0)
        
    #train_loss=torch.cat((train_loss,temp),0)
    del temp
    scheduler.step()
    
    with torch.no_grad():
        avgTestLoss=0
        for k,(x_test,y_test) in enumerate(test_DL):
            x_test=x_test.to(device)
            y_test=y_test.to(device)
            with torch.autocast(device_type="cuda"):
                ypred_val=model.forward(x_test)
                testLoss = lossF(ypred_val, y_test)
                avgTestLoss += testLoss
               
        temp=avgTestLoss/(k+1)
        if temp.dim()==0:
            temp=torch.tensor([temp]) 
            
        temp=temp.to(device)
        if i==0:
            test_loss=torch.tensor(temp)
        else:
            test_loss=torch.cat((test_loss,temp),0)
            
        del temp
        
        
    torch.save(model.state_dict(), './results/model_2DCNN_SST')
    lossTrain=train_loss.to('cpu').detach().numpy()
    lossTest=test_loss.to('cpu').detach().numpy()
    np.save('./results/trainHist.npy', np.array(lossTrain,dtype=object),allow_pickle=True)
    np.save('./results/testHist.npy', np.array(lossTest,dtype=object),allow_pickle=True)


    print(f'Epoch: {i}, train loss: {train_loss[i]}, test loss: {test_loss[i]}, time per epoch: {round(time()-epoch_start,3)}')
                 
total_time=time()-start
print(f'Total training time: {total_time/60}')

print('Saving training history')
lossTrain=train_loss.to('cpu').detach().numpy()
lossTest=test_loss.to('cpu').detach().numpy()
plt.plot(lossTrain,label='train')
plt.plot(lossTest,label='test')
plt.yscale('log')
plt.legend()
plt.savefig('./results/training.jpeg')

np.save('./results/trainHist.npy', np.array(lossTrain,dtype=object),allow_pickle=True)
np.save('./results/testHist.npy', np.array(lossTest,dtype=object),allow_pickle=True)

del lossTrain
del lossTest

print('Evaluating model')

with torch.no_grad():
    for k,(x_test,y_test) in enumerate(test_DL):
        x_test=x_test.to(device)
        y_test=y_test.to(device)
        if k==0:
            ypred_val=model.forward(x_test)
            ytest=y_test
        else:
            temp=model.forward(x_test)
            ypred_val=torch.cat([ypred_val,temp])
            ytest=torch.cat([ytest,y_test])
            
ypred_val=ypred_val.to('cpu')
ytest=ytest.to('cpu')
preds_np=np.array(ypred_val)
test_np=np.array(ytest)

error1=np.mean(abs(preds_np-test_np)/(test_np+1e-5))
error2=np.mean(abs(preds_np-test_np)/np.std(test_np,axis=0))
error3=np.mean(abs(preds_np-test_np)/np.max(test_np,axis=0))

print(f'Error1: {error1}, error2: {error2}, error3: {error3}')

corr=np.zeros((preds_np.shape[0]))
for i in range(len(preds_np)):
    temp1=np.sum(preds_np[i,0].flatten()*test_np[i,0].flatten())
    temp2=np.sqrt(np.sum(preds_np[i,0].flatten()**2))
    temp3=np.sqrt(np.sum(test_np[i,0].flatten()**2))
    corr[i]=temp1/(temp2*temp3)
    
print(f'Correlation coefficient: {np.mean(corr)}')

print('Saving model')
torch.save(model.state_dict(), './results/model_2DCNN_SST')












