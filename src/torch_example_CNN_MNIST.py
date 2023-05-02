

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torch import nn
from torch.autograd import Variable
from torch import optim
from torchvision import transforms


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.layer1 = nn.Sequential(  #输入 1 * 28 * 28
            nn.Conv2d(1,16,kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))  #第一层 输出 16* 26 *26

        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3),
            nn.BatchNorm2d(32), # 32 * 24 *24
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)) # 32 * 12* 12

        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3),
            nn.BatchNorm2d(64), # 64 * 10 * 10
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3),
            nn.BatchNorm2d(128), # 128 * 8 * 8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)) # 128 * 4 * 4

        self.fc = nn.Sequential(
                nn.Linear(128 * 4 * 4,1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024,128),
                nn.ReLU(inplace=True),
                nn.Linear(128,10))

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

data_tf = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize([0.5],[0.5])])

train_set = mnist.MNIST('./data',train=True,transform=data_tf,download=False)
test_set = mnist.MNIST('./data',train=False,transform=data_tf,download=False)

train_data = DataLoader(train_set,batch_size=64,shuffle=True)
test_data = DataLoader(test_set,batch_size=128,shuffle=False)

print("complete dataset loading \n")



net = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),1e-1)
nums_epoch = 10

losses =[]
acces = []
eval_losses = []
eval_acces = []

for epoch in range(nums_epoch):
    print("epoch :", epoch+1)
    #TRAIN
    train_loss = 0
    train_acc = 0
    net = net.train() #初始化网络
    num_batch = 0
    for img , label in train_data:
        num_batch += 1

        # simg = img.reshape(img.size(0),-1)
        # print("simg ,label ",simg,label)  # 一个batch 是64张图片

        img = Variable(img)
        # print(img[0].shape)  [1, 28, 28]  28 * 28 的
        label = Variable(label)

        # forward
        out = net(img) # 调用forward, 返回预测值
        loss = criterion(out,label)  #损失函数，计算损失
        # backward
        optimizer.zero_grad()
        loss.backward()  # 反向传播，更新参数
        optimizer.step()

        # loss
        train_loss += loss.item()
        # accuracy
        _,pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc
        # print("num_batch: ",num_batch)

    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))


    #TEST
    eval_loss = 0
    eval_acc = 0
    for img , label in test_data:
        #img = img.reshape(img.size(0),-1)
        img = Variable(img)
        label = Variable(label)

        out = net(img)
        loss = criterion(out,label)
        eval_loss += loss.item()
        _ , pred = out.max(1)
        num_correct = (pred==label).sum().item()
        acc = num_correct / img.shape[0]
        eval_acc += acc

    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))

    #PRINT IN EVERYEPOCH
    print('Epoch {} Train Loss {} Train  Accuracy {} Teat Loss {} Test Accuracy {}'.format(
        epoch+1, train_loss / len(train_data),train_acc / len(train_data), eval_loss / len(test_data), eval_acc / len(test_data)))

