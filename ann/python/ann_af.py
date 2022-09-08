import numpy as np
from random import random

# Author: Maurice Snoeren - Avans Hogeschool - mac.snoeren@avans.nl
# 09-2020 - Beta version

class ANN_Activation:
    """ Artificial Neural Network activation function class.
        This class implements the base class of the activation function. It implements the
        activation function and the derivative function of the activation function.
    """

    def __init__(self):
        """Constructor for the ANN_Activation."""
        pass

    def forward(self, input_vector):
        """Forward function of the activation function."""
        print("ABNN_Activation base class!")
        return input_vector

    def derivative(self, input_vector): 
        """Derivative of the function, used by the back propagation algorithm."""
        print("ABNN_Activation base class!")
        return input_vector        

class ANN_Sigmoid_Activation(ANN_Activation):
    """ Artificial Neural Network signoid activation function class.
    """

    def __init__(self):
        """Constructor for the ANN_Sigmoid_Activation."""
        pass

    def forward(self, input_vector):
        return 1/(1 + np.exp(-input_vector)) 

    def derivative(self, input_vector): 
        print("input: " + str(input_vector))
        return np.exp(-input_vector) / ((1 + np.exp(-input_vector))**2)

    def __str__(self):
        return "Sigmoid activation function"
