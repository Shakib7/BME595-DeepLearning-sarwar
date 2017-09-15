import math
import torch
from math import sqrt
import random
class NeuralNetwork:
    def __init__(self, list):
        self.list = list
        n = len(self.list)-1
        Theta=[]
        for i in range(n):
            Theta.append(torch.FloatTensor([[random.gauss(0,1/(sqrt(self.list[i+1]))) for x in range(self.list[i+1])] for y in range(self.list[i]+1)]))
        self.__Theta =Theta

    def getLayer(self,i):
        return self.__Theta[i]

    #def __call__(self,x,y):
    #    return self.forward(self,x,y)

    def forward (self,input):
        if (input.numpy()).ndim is 1:
            L=1
        else:
            L=len(input[0])
        self.L=L    #batch size

        ip = []; sg=[]; eg=[]
        listB=torch.IntTensor(self.list);   listB[len(self.list)-1]=listB[len(self.list)-1]-1
        for i in range(len(self.list)):
            ip.append(torch.FloatTensor([random.random() for x in range(listB[i]+1)]))
            sg.append(torch.FloatTensor([random.random() for x in range(listB[i]+1)]))
            eg.append(torch.FloatTensor([random.random() for x in range(listB[i]+1)]))
        self.temp = ip;     self.sigmGrad = sg;     self.ErrGrad=eg
        temp = input
        self.temp[0] = torch.cat({torch.ones(1, L), input}, 0)
        for i in range(len(self.list)-1):
            out = torch.cat({torch.ones(1, L), temp}, 0)
            self.temp[i] =out
            temp=torch.mm(torch.transpose(self.__Theta[i],0,1),out)

        # ~~~~~~~~~~~~~~~~~Sigmoid~~~~~~~~~~~~~~~~~~~#
            for k in range(L):
                for j in range(len(temp)):
                    temp[j][k]=1 / (1 + math.exp(-temp[j][k]))
            self.sigmGrad[i+1]=torch.addcmul(torch.zeros(len(temp),L),temp,(1-temp))
        self.temp[-1] = temp

        if self.L is 1:
            temp=temp[:,0]
        else:
            pass
        return temp


    def backward (self, target, loss="MSE"):

        #print ('loss',loss)

        n = len(self.list) - 1
        self.target = target
        dE_dTheta = []
        for i in range(n):
            dE_dTheta.append(torch.FloatTensor([[[random.gauss(0, 1 / (sqrt(self.list[i + 1]))) for x in range(self.list[i + 1])] for y in range(self.list[i] + 1)] for z in range(self.L)]))

        #Selecting action for different loss functions
        if loss is "MSE":
            self.ErrGrad[len(self.list) - 1] = (self.temp[len(self.list) - 1] - self.target)
            self.Err = 0.5 * (torch.pow(self.ErrGrad[len(self.list) - 1], 2))
            TotalErr = torch.sum(self.Err)
        else:
            pass

        #EGSG=errGradn x sigmGradn  n->output layer
        EGSG=torch.addcmul(torch.zeros(len(self.sigmGrad[len(self.list)-1]),self.L),self.ErrGrad[len(self.list)-1],self.sigmGrad[len(self.list)-1])
        for i in range(self.L):
            dE_dTheta[len(self.list) - 2][i]=torch.mm(torch.unsqueeze(self.temp[-2][:, i],1),torch.transpose(torch.unsqueeze(EGSG[:,i],1),0,1))


        for i in range((len(self.list) - 3),-1,-1):
            #xy=errGradn x sigmGradn
            xy=torch.transpose(torch.addcmul(torch.zeros(len(self.sigmGrad[i+2]),self.L),self.ErrGrad[i+2],self.sigmGrad[i+2]),0,1)
            # ErrGrad(n-1)=xy*theta(n)
            self.ErrGrad[i+1] = torch.mm(xy, torch.transpose(self.__Theta[i + 1][1:(self.list[i+1]+1), :], 0, 1))

            for j in range(self.L):
                #localInput*sigmoidGradient=sgxlip
                sgxlip=torch.mm(torch.unsqueeze(self.temp[i][:,j],1), torch.transpose(torch.unsqueeze(self.sigmGrad[i+1][:,j],1), 0, 1))
                ErrGrd = torch.unsqueeze(torch.FloatTensor(self.ErrGrad[i+1][j,:]),0)
                #ErrGrd in matrix form
                for k in range(len(self.temp[i][:,j]) - 1):
                    ErrGrd = torch.cat((ErrGrd, torch.unsqueeze(torch.FloatTensor(self.ErrGrad[i+1][j,:]),0)), 0)
                #dE_dTheta=ErrGrdxlocalInput*sigmoidGradient
                dE_dTheta[i][j] = torch.addcmul(torch.zeros(len(sgxlip), len(sgxlip[0])), sgxlip, ErrGrd)

        #average dE_dTheta
        de_dtheta = [];
        for i in range(len(self.list) - 1):
            de_dtheta.append(dE_dTheta[i][0])
        for i in range(len(self.list)-1):
            for j in range(1,self.L,1):
                de_dtheta[i] += dE_dTheta[i][j]
            de_dtheta[i] /= self.L
        self.dE_dTheta=de_dtheta
        return TotalErr

    def updateParams(self,eta):
        self.eta=eta
        for i in range(len(self.list)-1):
            self.__Theta[i]=self.__Theta[i] - eta * self.dE_dTheta[i]
        print ('updated Theta',self.__Theta)
