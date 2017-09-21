from __future__ import print_function
import random
import torch
import torchvision
from neural_network import NeuralNetwork

class MyImg2Num:
	def __init__(self):
		layers=[784,50,10]										#[in,h1,...,out]
		self.model = NeuralNetwork(layers)

	def __call__(self, img):
		return self.forward(img)

	def forward(self,img):
		return self.model.forward(img)

	def train(self):
	# ~~~~~~~~~~~~~~~~~Train Network~~~~~~~~~~~~~~~~~~~#
	#Prep Dataset
		train_loader = torchvision.datasets.MNIST('../data_MNIST', train=True, download=True, transform=True,target_transform=True)
		test_loader = torchvision.datasets.MNIST('../data_MNIST', train=False, download=True, transform=True,target_transform=True)

		epoch=2;	batch_size = 100;	numCls = 10;	iteration = len(train_loader) / batch_size;
		input=(torch.squeeze(train_loader.train_data.view(iteration, batch_size, 1, 784), 2)).type(torch.FloatTensor)
		target = torch.FloatTensor(len(train_loader.train_labels), numCls)

		##one hot encode
		for i in range(len(train_loader.train_labels)):
			target[i][train_loader.train_labels[i]] = 1
		target=target.view(iteration, batch_size,numCls)
	#epochs
		#Train
		for i in range(epoch):
			index=random.sample(range(iteration),iteration)
			e = 0
			with open("outLoss" + str(i) + ".csv", "w") as out_file:
				for j in range(iteration): #iteration
					self.model.forward(input[index[j]])
					e +=self.model.backward(target[index[j]])
					self.model.updateParams(0.5)
					print ('ij',i,j)
				print('Total Err (i)',i,':', e)
				out_file.write(str(e))
		#Test
			with open("accuracy" + str(i) + ".csv", "w") as acc_file:
				batch_size = 100;	numCls = 10;	iterationT = len(test_loader) / batch_size;
				inputT = (torch.squeeze(test_loader.test_data.view(iterationT, batch_size, 1, 784), 2)).type(torch.FloatTensor)
				targetT = torch.zeros(len(test_loader.test_labels), numCls)
				##one hot encode
				for i in range(len(test_loader.test_labels)):
					targetT[i][test_loader.test_labels[i]] = 1
				targetT = targetT.view(iterationT, batch_size, numCls)

				err = 0
				for i in range(iterationT):
					err += len(torch.nonzero(self.model.forward(inputT[i]) - targetT[i])) / 2
				print('acc', 100 - ((err * 100) / len(test_loader.test_labels)))
				acc_file.write(str(100 - ((err * 100) / len(test_loader.test_labels))))

	##Test After all epochs
		batch_size = 100;	numCls = 10;	iterationT = len(test_loader) / batch_size;
		inputT = (torch.squeeze(test_loader.test_data.view(iterationT, batch_size, 1, 784), 2)).type(torch.FloatTensor)
		targetT = torch.zeros(len(test_loader.test_labels), numCls)
		##one hot encode
		for i in range(len(test_loader.test_labels)):
			targetT[i][test_loader.test_labels[i]] = 1
		targetT = targetT.view(iterationT, batch_size, numCls)

		err=0
		for i in range(iterationT):
			err +=len(torch.nonzero(self.model.forward(inputT[i])-targetT[i]))/2
		print ('acc',100-((err*100)/len(test_loader.test_labels)))
