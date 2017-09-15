#HW03 - Neural Networks- Back-propagation pass :NeuralNetwork API and logic_gates API

###1. The theta values arrived from training were similar to HW02 values. It was just sclaed down version with similar ratios as HW02.
###2. Training Time for Iso-error - XOR > OR > AND > NOT
###3. test.py is included for testing the NeuralNetwork API and logic_gates API. 

## NeuralNetwork API has one class with five methods __init__, getLayer(), forward(), backward() and updateParams().
## logic_gates API has four classes AND, OR, NOT, XOR each with four methods __init__, __call__, forward() and train().
## Tried to implement Cross Entropy Loss function also. Could not understand the terms in the equation, therefore was unable to finish.

### NeuralNetwork API methods :-
1. __init__ -  initializes a neural network model of given structure - no. of layers, size of each layer. Returns a dictionary of theta matrices.
2. getLayer() - returns the theta matrix corresponding to **layer(i)** and **layer(i+1)**
3. forward() - propagates input across the neural network and returns the final output vector.
4. backward() - calculates loss and dE_dThetas. Averages the dE_dThetas for batch input.
5. updateParams() - updates the Thetas using eta*dE_dTheta.
* Works for single input vertor as well as batch inputs

### logic_gates API :- (takes boolean inputs and returns boolean output)
1. AND - logical **AND** of two inputs
2. OR - logical **OR** of two inputs
3. NOT - logical **NOT** of an inputs
4. XOR - logical **XOR** of two inputs (not linearly separable, therefore require more layers and more cycles than OR, AND and NOT gates)
* Works only for 2 inputs (1 for NOT gate) as number of input arguments are hard coded.

[GitHub](https://github.com/Shakib7/BME595-DeepLearning-sarwar)