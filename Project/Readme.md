# Incrimental Learning using modified ResNet
Will try to learn ImageNet dataset incrimentally with transfer learning.

## Team members
Syed Shakib Sarwar (Shakib7)

## Goals
1. Train whole ImageNet (1000 classes) using ResNet18 in PyTorch.
2. Train ImageNet classes separately in 3 batches (500,300,200) using ResNet18 in PyTorch.
3. Transfer weights from trained net with 500 classes and train only the classifier for 300 and 200 classes.
4. Merge the separately learned networks.
5. Test accuracy for the individual networks and the merged network.

## Challenges
1. Training ImageNet using PyTorch and GPU.
2. Transfering weights from trained network to another network.
3. Merging separately learned networks and testing accuracy.
