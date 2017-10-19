## HW06 - Transfer Learning
Used Python 2.7.14 and 4 GPUs

# Part1. Created model definition of AlexNet. 

Loaded pretrained AlexNet model for ImageNet. Copied parameters from pretrained model to my model. 

Replaced last Linear layer to work with 200 classes of Tiny-imagenet instead of 1000 classes of ImageNet.

Trained the last Linear Layer with Tiny-imagenet.

Trained for 50 epochs with starting learning rate 0.01.

Validation Accuracy achieved =~55% (top1), ~80%(top5).

# Source: train.py

# Train command: python train.py --data /tiny-imagenet-200/ --save saved_models/
 

# Part2. Initializes camera and then gets continuous feed from it. These frames are then sent to forward function to get the prediction. 
Based on the prediction, it shows the output overlayed on the input image. Couldn't get class captions. Used class number (0-199) as caption on the image.
Webcam sometimes doesn't work, therefore provision given inside code to test with stored images.

# Source: test.py

# Test command: python test.py --model saved_models/

[GitHub](https://github.com/Shakib7/BME595-DeepLearning-sarwar/tree/master/HW6)