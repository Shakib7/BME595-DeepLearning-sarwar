import torch
import random
from neural_network import NeuralNetwork

class AND:
    def __init__(self):
        layers=[2,1]
        self.a = NeuralNetwork(layers)
#        th0 = self.a.getLayer(0)
#        th0[0]=-30; th0[1] =20; th0[2] =20
    def __call__(self,x,y):
        return self.forward(x,y)
    def forward (self,x,y):
        return bool(self.a.forward(torch.FloatTensor([int(bool(x)),int(bool(y))]))[0]>0.5)
    def train(self):
        eta=0.5 #Learning rate
        # Generate data on the fly
        x=bool(random.randint(0,1)) ; y=bool(random.randint(0,1)); target=bool(x and y)
        self.a.forward(torch.FloatTensor([int(bool(x)), int(bool(y))]))[0]
        self.a.backward(torch.FloatTensor([int(bool(target))]))
        self.a.updateParams(eta)

class OR:
    def __init__(self):
        layers=[2,1]
        self.o = NeuralNetwork(layers)
#        th0=self.o.getLayer(0)
#        th0[0]=-10; th0[1] =20; th0[2] =20
    def __call__(self,x,y):
        return self.forward(x,y)
    def forward(self,x, y):
        return bool(self.o.forward(torch.FloatTensor([int(bool(x)),int(bool(y))]))[0]>0.5)
    def train(self):
        eta=0.5 #Learning rate
        # Generate data on the fly
        x=bool(random.randint(0,1)) ; y=bool(random.randint(0,1)); target=bool(x or y)
        self.o.forward(torch.FloatTensor([int(bool(x)), int(bool(y))]))[0]
        self.o.backward(torch.FloatTensor([int(bool(target))]))
        self.o.updateParams(eta)

class NOT:
    def __init__(self):
        layers=[1,1]
        self.n = NeuralNetwork(layers)
#        th0=self.n.getLayer(0)
#        th0[0]=10;  th0[1] =-20
    def __call__(self,x):
        return self.forward(x)
    def forward(self,x):
        return bool(self.n.forward(torch.FloatTensor([int(bool(x))]))[0]>0.5)
    def train(self):
        eta=0.5 #Learning rate
        # Generate data on the fly
        x=bool(random.randint(0,1));    target=bool(not x)
        self.n.forward(torch.FloatTensor([int(bool(x))]))[0]
        self.n.backward(torch.FloatTensor([int(bool(target))]))
        self.n.updateParams(eta)

class XOR:
    def __init__(self):
        layers=[2,2,1]
        self.X = NeuralNetwork(layers)
#        th0=self.X.getLayer(0)
#        th0[0]=-10; th0[1] =20; th0[2] =20
    def __call__(self,x,y):
        return self.forward(x,y)
    def forward(self,x, y):
        return bool(self.X.forward(torch.FloatTensor([int(bool(x)),int(bool(y))]))[0]>0.5)
    def train(self):
        eta=0.5 #Learning rate
        # Generate data on the fly
        x=bool(random.randint(0,1)) ; y=bool(random.randint(0,1)); target=bool(((x or y) and (not (x and y))))
        self.X.forward(torch.FloatTensor([int(bool(x)), int(bool(y))]))[0]
        self.X.backward(torch.FloatTensor([int(bool(target))]))
        self.X.updateParams(eta)