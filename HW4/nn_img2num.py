from __future__ import print_function
import random
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torch.optim as optim


class NnImg2Num:
    def __init__(self):
        layers = [784, 50, 10]  # [in,h1,...,out]
        self.model = torch.nn.Sequential(
            torch.nn.Linear(784, 50),
            torch.nn.Sigmoid(),
            torch.nn.Linear(50, 10),
            torch.nn.Sigmoid(),
        )

    def __call__(self, img):
        return self.forward(img)

    def forward(self, x):
        tout=(self.model.forward(Variable(x))).data
        tout, index = torch.max(tout, 1)
        predict = torch.zeros(len(index), 10)
        ##one hot encode
        for i in range(len(index)):
            predict[i][index[i]] = 1
        return predict

    def train(self):
        loss_fn = torch.nn.MSELoss(size_average=False)
        learning_rate = 0.001
        optimizer = optim.SGD(self.model.parameters(),lr=learning_rate)

        # ~~~~~~~~~~~~~~~~~Train Network~~~~~~~~~~~~~~~~~~~#
        # Prep Dataset
        train_loader = torchvision.datasets.MNIST('../data_MNIST', train=True, download=True, transform=True,
                                                  target_transform=True)
        test_loader = torchvision.datasets.MNIST('../data_MNIST', train=False, download=True, transform=True,
                                                 target_transform=True)
        epoch = 1;  batch_size = 100;   numCls = 10;    iteration = len(train_loader) / batch_size
        input = (torch.squeeze(train_loader.train_data.view(iteration, batch_size, 1, 784), 2)).type(torch.FloatTensor)
        target = torch.FloatTensor(len(train_loader.train_labels), numCls)

        ##one hot encode
        for i in range(len(train_loader.train_labels)):
            target[i][train_loader.train_labels[i]] = 1
        target = Variable(target.view(iteration, batch_size, numCls))

        # epochs
        #train
        for i in range(epoch):
            index = random.sample(range(iteration), iteration)
            e = 0
            with open("outLoss"+str(i)+".csv", "w") as out_file:
                for j in range(iteration):
                    optimizer.zero_grad()
                    output=self.model.forward(Variable (input[index[j]]))
                    loss = loss_fn(output, target[index[j]])
                    e +=loss.data[0]
                    #print ('loss(ij)',i,j,':',loss.data[0] )
                    loss.backward()
                    optimizer.step()
                print ('loss(i)',i,':',e)
                out_file.write(str(e))


        #~~~~~~~~~~~~~~~~~Test Network~~~~~~~~~~~~~~~~~~~#
        batch_size = 100;	numCls = 10;	iterationT = len(test_loader) / batch_size;
        inputT = (torch.squeeze(test_loader.test_data.view(iterationT, batch_size, 1, 784), 2)).type(torch.FloatTensor)
        targetT = torch.zeros(len(test_loader.test_labels), numCls)
        ##one hot encode
        for i in range(len(test_loader.test_labels)):
            targetT[i][test_loader.test_labels[i]] = 1
        targetT = Variable(targetT.view(iterationT, batch_size, numCls))
        err=0
        for i in range(iterationT):   #iteration
            outputT=self.forward(inputT[i])
            err += len(torch.nonzero(outputT - targetT[i].data)) / 2
        print ('acc',100-((err*100)/len(test_loader.test_labels)))
