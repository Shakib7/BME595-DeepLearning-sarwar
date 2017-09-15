from __future__ import print_function
from logic_gates import AND
from logic_gates import OR
from logic_gates import NOT
from logic_gates import XOR
from neural_network import NeuralNetwork
import numpy
import torch
import math

def test():
	# ~~~~~~~~~~~~~~~~~Test Neural Network with batch input~~~~~~~~~~~~~~~~~~~#
	layers = [3, 4, 2]  # [in,h1,...,out]
	print('layers', layers)
	model = NeuralNetwork(layers)  # initialize neural network
	# input = torch.DoubleTensor([[1, 3, 1], [2, 2, 2]])	#2D DoubleTensor
	input = torch.DoubleTensor([1, 3, 1])  # 1D DoubleTensor
	if (input.numpy()).ndim is 1:
		pass
	else:
		input = torch.transpose(input, 0, 1)
	print('input', input)
	print('output', model.forward(input))
	# ~~~~~~~~~~~~~~~~~Test getLayer method~~~~~~~~~~~~~~~~~~~#
	th0 = model.getLayer(0)
	print('th0=', th0)
	# th0[0]=-20; 	th0[1] =15;		#th0[2] =15		#modify theta
	# print('th0m=', th0)

	# ~~~~~~~~~~~~~~~~~Test Logic Gates~~~~~~~~~~~~~~~~~~~#
	And=AND()
	print(And(False, False));	print (And(False, True));	print (And(True, False));	print (And(True, True))
	Or = OR()
	print(Or(False, False));	print(Or(False, True));		print(Or(True, False));		print(Or(True, True))
	Not = NOT()
	print(Not(False));			print(Not(True))
	Xor = XOR()
	print(Xor(False, False));	print(Xor(False, True));	print(Xor(True, False));	print(Xor(True, True))

test()
	
