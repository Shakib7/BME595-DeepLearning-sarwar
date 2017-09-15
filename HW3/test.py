from __future__ import print_function
from logic_gates import AND
from logic_gates import OR
from logic_gates import NOT
from logic_gates import XOR
from neural_network import NeuralNetwork
import numpy
import torch
#import torchvision
import math

def test():
	#~~~~~~~~~~~~~~~~~Test Neural Network with batch input~~~~~~~~~~~~~~~~~~~#
	layers=[2,2,2]										#[in,h1,...,out]
	#print('layers',layers)
	model=NeuralNetwork(layers)							#initialize neural network
	## single input
	#input = torch.FloatTensor([0.05, 0.1]);	target= torch.FloatTensor([0.01, 0.99])
	## bacth input
	input = torch.FloatTensor([[0.05, 0.1], [0.05, 0.1]]);	target = torch.FloatTensor([[0.01, 0.99],[0.01, 0.99]])
	if (input.numpy()).ndim is 1:
		pass
	else:
		input = torch.transpose(input, 0, 1)
		target = torch.transpose(target, 0, 1)
	#print('input',input)
	#print('target', target)
	th0 = model.getLayer(0)
	#print('th0=', th0)
	th0[0][0]=0.35; 	th0[1][0] =0.15;		th0[2][0] =0.2		#modify theta
	th0[0][1] = 0.35;	th0[1][1] = 0.25;		th0[2][1] = 0.3  # modify theta
	#print('th0m=', th0)
	th1 = model.getLayer(1)
	#print('th1=', th1)
	th1[0][0] = 0.6;	th1[1][0] = 0.4;		th1[2][0] = 0.45  # modify theta
	th1[0][1] = 0.6;	th1[1][1] = 0.5;		th1[2][1] = 0.55  # modify theta
	#print('th1m=', th1)
	print('output',model.forward(input))
	#print('Total Err', model.backward(target))
	#model.updateParams(0.5)
	# ~~~~~~~~~~~~~~~~~Train Network~~~~~~~~~~~~~~~~~~~#
	for i in range(100):
		print('output', model.forward(input))
		print('Total Err', model.backward(target))
		model.updateParams(0.5)


	# ~~~~~~~~~~~~~~~~~Test Logic Gates~~~~~~~~~~~~~~~~~~~#
	#And=AND()
	##print(And(False, False));	print (And(False, True));	print (And(True, False));	print (And(True, True))
	#for i in range(500):
	#	And.train()
	#print(And(False, False));	print(And(False, True));	print(And(True, False));		print(And(True, True))

	#Or = OR()
	##print(Or(False, False));	print(Or(False, True));		print(Or(True, False));		print(Or(True, True))
	#for i in range(500):
	#	Or.train()
	#print(Or(False, False));	print(Or(False, True));		print(Or(True, False));		print(Or(True, True))

	#Not = NOT()
	##print(Not(False));			print(Not(True))
	#for i in range(500):
	#	Not.train()
	#print(Not(False));			print(Not(True))

	Xor = XOR()
	##print(Xor(False, False));	print(Xor(False, True));	print(Xor(True, False));	print(Xor(True, True))
	for i in range(50000):
		Xor.train()
	print(Xor(False, False));	print(Xor(False, True));	print(Xor(True, False));	print(Xor(True, True))

test()
	
