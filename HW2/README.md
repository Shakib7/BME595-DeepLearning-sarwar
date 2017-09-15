#HW02 - NeuralNetwork API and logic_gates API

## NeuralNetwork API has one class with three methods __init__, getLayer() and forward().
## logic_gates API has four classes AND, OR, NOT, XOR each with three methods __init__, __call__ and forward().

### NeuralNetwork API methods :-
1. __init__ -  initializes a neural network model of given structure - no. of layers, size of each layer. Returns a dictionary of theta matrices.
2. getLayer() - returns the theta matrix corresponding to **layer(i)** and **layer(i+1)**
3. forward() - propagates input across the neural network and returns the final output vector.
* Works for single input vertor as well as batch inputs

### logic_gates API :- (takes boolean inputs and returns boolean output)
1. AND - logical **AND** of two inputs
2. OR - logical **OR** of two inputs
3. NOT - logical **NOT** of an inputs
4. XOR - logical **XOR** of two inputs (implemented using neural networks of OR, AND and NOT gates)
* Works only for 2 inputs (1 for NOT gate) as number of input arguments are hard coded.

[GitHub](https://github.com/Shakib7/BME595-DeepLearning-sarwar)