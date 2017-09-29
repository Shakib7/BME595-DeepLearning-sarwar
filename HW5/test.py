from __future__ import print_function
from img2num import Img2Num
from img2obj import Img2Obj
import numpy as np
import random
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable


def test():
	#### prepare sample input for forward() in MNIST, need to unquote the 3 lines below
	'''test_loader = torchvision.datasets.MNIST('../data_MNIST', train=False, download=True, transform=True,target_transform=True)
	batch_size = 100;	numCls = 10;	iterationT = len(test_loader) / batch_size;
	input = (test_loader.test_data.view(iterationT, batch_size, 1,28, 28)).type(torch.FloatTensor)
	'''

	#### prepare sample input for forward() in CIFAR100, need to unquote the 4 lines below
	'''test_loader = torchvision.datasets.CIFAR100('../data_CIFAR100', train=False, download=True, transform=True,target_transform=True)
	batch_size = 100;	numCls = 100;	iterationT = len(test_loader) / batch_size
	input = (torch.from_numpy(test_loader.test_data)).type(torch.FloatTensor)
	input=(torch.transpose(input.view(iterationT, batch_size, 32, 32,3),2,4))#/255
	'''

	###### Test Img2Obj
	#net = Img2Obj()
	#net = torch.load('c_l.75_500')			# load trained net
	#print(net(input[0][13]))  				# test inference for one image [32x32] byte tensor
	#print(net.view(input[0][13]))			# test view function
	#print (test_loader.test_labels[13])	# test label
	#net.cam()								# test cam(), press 'q' to stop inference
	#net.train()							# train network

	###### Test Img2Num
	#net = Img2Num()
	#net = torch.load('MNIST_05_40')			# load trained net
	#print(net(input[0][2][0]))  			# test inference for one image [28x28] byte tensor
	#print (test_loader.test_labels[2])		# test label
	#net.train()							# train network
	#torch.save(net, 'MNIST_05_40')

test()
