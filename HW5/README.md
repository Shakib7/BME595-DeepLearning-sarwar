## HW05 - Neural Networks- CNN in PyTorch

#Part1. MNIST training in LeNet-5 converged in ~40 epochs. MNIST training in FCN-[784 50 10] converged in ~35 epochs. 
Total training time for LeNet-5= 436s(for 40 epochs), while total training time for FCN-[784 50 10]= 35s(for 35 epochs).
Inference time for LeNet-5= 0.5s(per image), while inference time for FCN-[784 50 10]= ~0.3s(per image).
Test Accuracy for LeNet-5=99%, while for FCN-[784 50 10]=93%.

#Part2. Accuracy achieved by LeNet-5 trained on CIFAR-100 is ~27%. Converged in 750 epochs. 
Total training time ~9500s. Inference time = 0.5s(per image)

##Saved nets:
1. c_l.75_500: Trained LeNet-5 on CIFAR-100 (500 epochs)
2. MNIST_05_40: Trained LeNet-5 on MNIST	(40 epochs)

### Img2Num API methods :- (for MNIST)
1. __init__ -  initializes a neural network model of given structure - no. of layers, size of each layer. 
2. __call__ - calls forward() method.
3. forward() - propagates input across the neural network and returns the final output vector.
4. train() - trains the network using back-propagation.

### Img2Obj API :- (for CIFAR-100)
1. __init__ -  initializes a neural network model of given structure - no. of layers, size of each layer. 
2. __call__ - calls forward() method.
3. forward() - propagates input across the neural network and returns predicted caption.
4. train() - trains the network using back-propagation.
5. view() - propagates input across the neural network and returns predicted caption overlayed on the input image.
6. cam() - initializes camera and then gets continuous feed from it. These frames are then sent to forward function to get the prediction. Based on the prediction, it shows the output overlayed on the input image.

[GitHub](https://github.com/Shakib7/BME595-DeepLearning-sarwar/tree/master/HW5)

![Accuracy vs epoch (LeNet-5 and FCN trained on MNIST)](https://github.com/Shakib7/BME595-DeepLearning-sarwar/blob/master/HW5/MNIST_Accuracy.pdf)
![Time vs epoch (LeNet-5 and FCN trained on MNIST)](https://github.com/Shakib7/BME595-DeepLearning-sarwar/blob/master/HW5/MNIST_Time.pdf
![Accuracy vs epoch (LeNet-5 trained on CIFAR-100)](https://github.com/Shakib7/BME595-DeepLearning-sarwar/blob/master/HW5/CIFAR_Accuracy.pdf)
![Time vs epoch (LeNet-5 trained on CIFAR-100)](https://github.com/Shakib7/BME595-DeepLearning-sarwar/blob/master/HW5/CIFAR_Time.pdf)