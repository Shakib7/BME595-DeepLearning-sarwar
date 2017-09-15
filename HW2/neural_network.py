import math
import torch
from math import sqrt
#import numpy as np
import random

class NeuralNetwork:
    def __init__(self, list):
        self.list = list
        n = len(self.list)-1
        #i=0
        theta=[]
        for i in range(n):
           theta.append(torch.DoubleTensor([[random.gauss(0,1/(sqrt(self.list[i+1]))) for x in range(self.list[i+1])] for y in range(self.list[i]+1)]))
        self.__theta =theta

    def getLayer(self,i):
        return self.__theta[i]

    #def __call__(self,x,y):
    #    return self.forward(self,x,y)

    def forward (self,input):
        #i=0;L=0
        temp = input.type(torch.FloatTensor)
        if (input.numpy()).ndim is 1:
            L=1
        else:
            L=len(temp[0])
        for i in range(len(self.list)-1):
            out = torch.cat({torch.ones(1, L), temp}, 0)
            out = out.type(torch.DoubleTensor)
            #print (self.__theta[i])
            temp=torch.mm(torch.transpose(self.__theta[i],0,1),out)
            temp = temp.type(torch.FloatTensor)
            #print ('temp', temp)
            #j=0;k=0
        # ~~~~~~~~~~~~~~~~~Sigmoid~~~~~~~~~~~~~~~~~~~#
            for k in range(L):
                for j in range(len(temp)):
                    temp[j][k]=1 / (1 + math.exp(-temp[j][k]))
            #print ('SigTemp', temp)

        temp = temp.type(torch.DoubleTensor)
        if L is 1:
            temp=temp[:,0]
        else:
            pass
        return temp

