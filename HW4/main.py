from __future__ import print_function
from neural_network import NeuralNetwork
from nn_img2num import NnImg2Num
from my_img2num import MyImg2Num
import numpy
import random
import torch
import torchvision
from torch.autograd import Variable


def main():
	# prepare sample input for forwad()
	train_loader = torchvision.datasets.MNIST('../data_MNIST', train=True, download=True, transform=True,
											  target_transform=True)
	epoch = 10;
	learning_rate = 0.5;
	batch_size = 100;
	numCls = 10;
	iteration = len(train_loader) / batch_size;
	input = (torch.squeeze(train_loader.train_data.view(iteration, batch_size, 1, 784), 2)).type(torch.FloatTensor)

	# Test NnImg2Num
	net = NnImg2Num()
	net.train()
	print('forward output', net.forward(input[0]))
	print('call output', net(input[0]))

	# Test MyImg2Num
	netMy = MyImg2Num();
	netMy.train()
	print('forward output', netMy.forward((input[0])))
	print('call output', netMy((input[0])))

main()
