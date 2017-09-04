from __future__ import print_function
from conv import Conv2D
from time import time
import matplotlib.image as img
import cv2
import matplotlib.pylab as plt
import numpy
import torch
import torchvision
from PIL import Image
import math
def main():
    t=1     #select Task
    if (t==1):
        in_channel=3
        o_channel=1
        kernel_size=3
        stride=1
        mode='known'
    elif (t==2):
        in_channel = 3
        o_channel = 2
        kernel_size = 5
        stride = 1
        mode = 'known'
    elif (t==3):
        in_channel = 3
        o_channel = 3
        kernel_size = 3
        stride = 2
        mode = 'known'

    conv2D = Conv2D(in_channel, o_channel, kernel_size, stride)

    #########################    read image #################################
    a = cv2.imread("1280x720.jpg")
    #print(len(a), len(a[0]), len(a[0][0]))
    #a = numpy.ndarray.reshape(a, (3, 1920, 1080))
    a = numpy.ndarray.reshape(a, (3, 1280, 720))
    #print(len(a), len(a[0]), len(a[0][0]))
    ##############################

    [ops,d] = conv2D.forward(a)
    print (ops,d)



main()
