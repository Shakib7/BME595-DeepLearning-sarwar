#import math
import torch
#import numpy as np
from neural_network import NeuralNetwork

class AND:
    def __init__(self):
        layers=[2,1]
        self.nn = NeuralNetwork(layers)
        th0 = self.nn.getLayer(0)
        th0[0]=-30; th0[1] =20; th0[2] =20
    def __call__(self,x,y):
        return self.forward(x,y)
    def forward (self,x,y):
        return bool(self.nn.forward(torch.DoubleTensor([int(bool(x)),int(bool(y))]))[0]>0.5)

class OR:
    def __init__(self):
        layers=[2,1]
        self.nn = NeuralNetwork(layers)
        th0=self.nn.getLayer(0)
        th0[0]=-10; th0[1] =20; th0[2] =20
    def __call__(self,x,y):
        return self.forward(x,y)
    def forward(self,x, y):
        return bool(self.nn.forward(torch.DoubleTensor([int(bool(x)),int(bool(y))]))[0]>0.5)

class NOT:
    def __init__(self):
        layers=[1,1]
        self.nn = NeuralNetwork(layers)
        th0=self.nn.getLayer(0)
        th0[0]=10;  th0[1] =-20
    def __call__(self,x):
        return self.forward(x)
    def forward(self,x):
        return bool(self.nn.forward(torch.DoubleTensor([int(bool(x))]))[0]>0.5)

class XOR:
    def __init__(self):
        self.nnOR = OR()
        self.nnAND = AND()
        self.nnNOT = NOT()
    def __call__(self,x,y):
        return self.forward(x,y)
    def forward(self,x, y):
        or_temp=self.nnOR(x,y)
        and_temp = self.nnAND(x, y)
        nand_temp = self.nnNOT(and_temp)
        return bool(self.nnAND(or_temp, nand_temp)>0.5)