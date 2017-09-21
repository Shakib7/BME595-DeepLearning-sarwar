### HW04 - Neural Networks- Back-propagation pass :NeuralNetwork API, MyImg2Num and NnImg2Num API

## 1. Network size is [784,50,10]

## 2. Training Time with my NeuralNetwork API is ~2000x compared to nn package

## 3. Accuracy is close. Though learning rate is different: 0.5 for MyImg2Num  and 0.001 for  NnImg2Num.

# NeuralNetwork API has one class with five methods __init__, getLayer(), forward(), backward() and updateParams().

# MyImg2Num API and NnImg2Num API each has one class with four methods __init__, __call__, forward() and train().

# The error vs epoch, accuracy vs epoch and time vs epoch charts have 5 points each. Could not run more epochs as my NeuralNetwork API takes too much time per epoch.

### MyImg2Num API methods :- (uses NeuralNetwork API)
1. __init__ -  initializes a neural network model of given structure - no. of layers, size of each layer. 
2. __call__ - calls forward() method.
3. forward() - propagates input across the neural network and returns the final output vector.
4. train() - trains the network using back-propagation.
* Works for single input vertor as well as batch inputs

### NnImg2Num API :- (uses nn package)
1. __init__ -  initializes a neural network model of given structure - no. of layers, size of each layer. 
2. __call__ - calls forward() method.
3. forward() - propagates input across the neural network and returns the final output vector.
4. train() - trains the network using back-propagation.
* Works for single input vertor as well as batch inputs

[GitHub](https://github.com/Shakib7/BME595-DeepLearning-sarwar/tree/master/HW4)

![Accuracy vs epoch (My)](https://github.com/Shakib7/BME595-DeepLearning-sarwar/blob/master/HW4/Acc(My).jpg)

![Accuracy vs epoch (NN)](https://github.com/Shakib7/BME595-DeepLearning-sarwar/blob/master/HW4/Acc(NN).jpg)

![Time vs epoch (My)](https://github.com/Shakib7/BME595-DeepLearning-sarwar/blob/master/HW4/Time(My).jpg)

![Time vs epoch (NN)](https://github.com/Shakib7/BME595-DeepLearning-sarwar/blob/master/HW4/Time(NN).jpg)

![Loss vs epoch (My)](https://github.com/Shakib7/BME595-DeepLearning-sarwar/blob/master/HW4/Loss(My).jpg)

![Loss vs epoch (NN)](https://github.com/Shakib7/BME595-DeepLearning-sarwar/blob/master/HW4/Loss(NN).jpg)