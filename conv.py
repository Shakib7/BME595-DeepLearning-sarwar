import math
import torch
import numpy
import cv2
import random
from time import time
class Conv2D:
	def __init__(self,in_channel,o_channel, kernel_size, stride):
		self.in_channel=in_channel
		self.o_channel=o_channel
		self.kernel_size=kernel_size
		self.stride=stride
		#self.mode=mode
	def forward(self, input):
		if (self.kernel_size==3):
			k = [[[-1, -1, -1], [0, 0, 0], [1, 1, 1]],[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]
		elif (self.kernel_size==5):
			k=[[[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1],[0,0,0,0,0],[1,1,1,1,1],[1,1,1,1,1]],[[-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1],[-1, -1, 0, 1, 1]]]
		else:
			k = numpy.array([[[random.random() for x in range(self.kernel_size + 0)] for y in range(self.kernel_size + 0)] for z in range(self.o_channel + 0)])
		p=(len(input[0])-self.kernel_size)/self.stride+1
		q=(len(input[0][0])-self.kernel_size)/self.stride+1
		r=self.in_channel
		s=self.o_channel
		print (r,s,p,q)
##################### Convolution #############################
		#d = torch.FloatTensor([[[0 for x in range(q+0)] for y in range(p+0)]for z in range(s+0)])
		d = numpy.array([[[0 for x in range(q + 0)] for y in range(p + 0)] for z in range(s + 0)])
		for o in range(s):
			for l in range(r):
				for m in range(p):
					for n in range(q):
						for i in range(self.kernel_size):
							for j in range(self.kernel_size):
								d[o][m][n] += input[l][i + m*self.stride][j + n*self.stride] * k[0][self.kernel_size-i-1][self.kernel_size-j-1]
		ops =math.pow(self.kernel_size,2)*p*q*self.in_channel*self.o_channel
########################## Normalization ############
		flat=numpy.ndarray.reshape(d,s*p*q)
		min=numpy.min(flat)
		flat=flat-min
		mx = numpy.max(flat)
		flat = numpy.array(flat, dtype=float)
		flat = numpy.array(flat/mx*255,dtype=float)
############################Save image####################################
#		a = numpy.ndarray.reshape(flat,( s, q, p))
#		a1=a[0,:,:]
#		a1=numpy.ndarray.reshape(a1, (q,p))
#		cv2.imwrite('N_T2_k4_1920.jpg', a1)
#		a2 = a[1, :, :]
#		a2 = numpy.ndarray.reshape(a2, (q, p))
#		cv2.imwrite('N_T2_k5_1920.jpg', a2)
	#	a3 = a[2, :, :]
	#	a3 = numpy.ndarray.reshape(a3, (q, p))
	#	cv2.imwrite('N_T3_k3_1280.jpg', a3)
		#print('d1',d1)
#		d0 = numpy.ndarray.reshape(d, (s, p, q))
#		d1 = d0[0,:, :]
#		d1 = numpy.ndarray.reshape(d1, (q, p))
#		cv2.imwrite('T2_k4_1920.jpg', d1)
#		d2 = d0[1, :, :]
#		d2 = numpy.ndarray.reshape(d2, (q, p))
#		cv2.imwrite('T2_k5_1920.jpg', d2)
	#	d3 = d0[2, :, :]
	#	d3 = numpy.ndarray.reshape(d3, (q, p))
	#	cv2.imwrite('T3_k3_1280.jpg', d3)

		d = torch.from_numpy(d)
		d = d.type(torch.FloatTensor)
		return (ops,d)