from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim


class Img2Num(nn.Module):
    def __init__(self):
        super(Img2Num,self).__init__()
        self.conv1=nn.Conv2d(1,6,5)
        self.conv2 =nn.Conv2d(6, 16, 5)
        self.fc1 =nn.Linear(16*4*4, 120)
        self.fc2 =nn.Linear(120, 84)
        self.fc3 =nn.Linear(84, 10)
        self.criterion=nn.MSELoss(size_average=True)

    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        img = (img.type(torch.FloatTensor)) / 255
        if (img.numpy()).ndim is 2:
            x = Variable(img.view(1,1,len(img),len(img[0])))
        elif (img.numpy()).ndim is 1:
            x = Variable(torch.unsqueeze(img, 0))
        else:
            x = Variable(img)
        #print (x)
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        ###convert to feature vector
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        x = x.view(-1, num_features)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train(self):
        learning_rate = 0.05
        optimizer = optim.SGD(self.parameters(),lr=learning_rate)

        # ~~~~~~~~~~~~~~~~~Train Network~~~~~~~~~~~~~~~~~~~#
        # Prep Dataset
        train_loader = torchvision.datasets.MNIST('../data_MNIST', train=True, download=True, transform=True,
                                                  target_transform=True)
        test_loader = torchvision.datasets.MNIST('../data_MNIST', train=False, download=True, transform=True,
                                                 target_transform=True)
        epoch = 40;  batch_size = 100;   numCls = 10;    iteration = len(train_loader) / batch_size
        input = ((train_loader.train_data.view(iteration, batch_size, 1,28, 28))) #.type(torch.FloatTensor))/255
        target = torch.zeros((len(train_loader.train_labels), numCls))

        ##one hot encode
        for i in range(len(train_loader.train_labels)):
            target[i][train_loader.train_labels[i]] = 1
        target = Variable(target.view(iteration, batch_size, numCls))

        #### epochs
        ###train
        for i in range(epoch):
            index = random.sample(range(iteration), iteration)
            e = 0
            for j in range(iteration):
                optimizer.zero_grad()
                output=self.forward(input[index[j]])
                #loss = loss_fn(output, target[index[j]])
                loss=self.criterion(output, target[index[j]])
                #print ('loss',loss)
                e +=loss.data[0]
                #print ('loss(ij)',i,j,':',loss.data[0] )
                loss.backward()
                optimizer.step()
            print ('loss(i)',i,':',e)


        #~~~~~~~~~~~~~~~~~Test Network~~~~~~~~~~~~~~~~~~~#
        batch_size = 100;	numCls = 10;	iterationT = len(test_loader) / batch_size
        inputT = ((test_loader.test_data.view(iterationT, batch_size, 1, 28,28))) #.type(torch.FloatTensor))/255
        targetT = torch.zeros((len(test_loader.test_labels), numCls))

        ##one hot encode
        for i in range(len(test_loader.test_labels)):
            targetT[i][test_loader.test_labels[i]] = 1
        targetT = Variable(targetT.view(iterationT, batch_size, numCls))
        err=0
        for i in range(iterationT):   #iterationT
            outputT=self.forward(inputT[i])
            #print ('O',outputT)
            tout, index = torch.max(outputT.data, 1)
            predict = torch.zeros(len(index), 10)
            ##one hot encode
            for j in range(len(index)):
                predict[j][index[j]] = 1
            outputT =predict
            #print('O', outputT)
            #print ('T',targetT[i].data)
            err += len(torch.nonzero(outputT - targetT[i].data)) / 2
        print ('acc',100-((err*100)/len(test_loader.test_labels)))
