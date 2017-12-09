# Incremental Learning using modified ResNet

## Team members
Syed Shakib Sarwar (Shakib7)

## Files
1. main.py: Train whole ImageNet (1000 classes) using ResNet34 in PyTorch.
command: python main.py /dataFolder
(dataFolder should have 1000 classes)
2. trainBaseModel.py: Train 500 ImageNet classes to get the base network.
command: python trainBaseModel.py /dataFolder 
(dataFolder should have 500 classes)
3. train_Incrementally.py: Train separately in 2 batches (300,200) using ResNet34 and Test accuracy for the individual networks.
command: python train_Incrementally.py /dataFolder 
(dataFolder should have 300 or 200 classes)
4. predict.py: For the individual networks, save prediction probabilities.
command: python predict.py /savedModelFolder 
5. ensemble.py: Merge the separately learned piction probabilities and Test accuracy for the merged network.
command: python ensemble.py /savedPredicitonFolder