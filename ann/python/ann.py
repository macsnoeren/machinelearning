import numpy as np
import matplotlib as plt
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
        return np.exp(-input_vector) / ((1 + np.exp(-input_vector))**2)

    def __str__(self):
        return "Sigmoid activation function"

class ANN_Hidden_Layer:
    """ Artificial Neural Network Hidden Layer class.
    """

    def __init__(self, num_input_nodes, num_hidden_nodes, activation):
        """Constructor for the ANN. 
            num_input_nodes (int): Number of inputs
            num_output_nodes (int): Number of outputs
        """
        self.num_input_nodes  = num_input_nodes
        self.num_hidden_nodes = num_hidden_nodes
        self.activation       = activation
        self.Wh               = np.random.rand(num_input_nodes, num_hidden_nodes) # Hidden Weight Matrix
        self.bh               = np.random.rand(num_hidden_nodes)                  # Biases vector
        self.h                = [] # Hold the output vector of this hidden layer

    def get_weight_matrix(self):
        return self.Wh

    def set_weight_matrix(self, Wh):
        self.Wh = Wh

    def get_biases_vector(self):
        return self.bh

    def set_biases_vector(self, bh):
        self.bh = bh

    def forward_propagation(self, x):
        """Calculate the output vector y of the neural network based on the x and hidden layers."""
        self.h = self.activation.forward( np.dot(x, self.Wh) + self.bh )
        return self.h

    def __str__(self):
        info = "(inputs: " + str(self.num_input_nodes) + ", nodes: " + str(self.num_hidden_nodes) + ", activation: " + str(self.activation) + ")\n"
        info += "  Weight Matrix: \n" + str(self.Wh) + "\n"
        info += "  Biases Vector: \n" + str(self.bh) + "\n"
        return info

class ANN:
    """ Artificial Neural Network class.
        This class implements a ANN with hidden layers to understand how a network is working under the hood.
    """

    def __init__(self, num_input_nodes, num_output_nodes, output_activation):
        """Constructor for the ANN. 
            num_input_nodes (int): Number of inputs
            num_output_nodes (int): Number of outputs
        """
        self.num_input_nodes   = num_input_nodes
        self.num_output_nodes  = num_output_nodes
        self.output_activation = output_activation
        self.hidden_layers     = [] # Holds the hidden_layer classes
        self.Wy                = np.random.rand(num_input_nodes, num_output_nodes) # Hold output layer weight matrix
        self.by                = np.random.rand(num_output_nodes) # Biases vector of the output nodes
        self.x                 = [] # Hold the input vector
        self.y                 = [] # hold the output vector

    def get_weight_matrix(self):
        return self.Wy

    def set_weight_matrix(self, Wy):
        self.Wy = Wy

    def get_biases_vector(self):
        return self.by

    def set_biases_vector(self, by):
        self.by = by

    def add_hidden_layer(self, hidden_layer):
        self.hidden_layers.append(hidden_layer)
        self.Wy = np.random.rand(hidden_layer.num_hidden_nodes, self.num_output_nodes)

    def forward_propagation(self, x):
        """Calculate the output vector y of the neural network based on the x and hidden layers."""
        self.x = x
        print("Input layer vector: " + str(x))
        total_hidden_layers = len(self.hidden_layers)

        if (total_hidden_layers == 0):
            self.y = self.output_activation.forward( np.dot( self.x, self.Wy ) + self.by )

        else: # Loop over the hidden layers
            input_vector = self.x
            for i in range( len(self.hidden_layers) ):
                output_vector = self.hidden_layers[i].forward_propagation(input_vector)
                print("Hidden layer output vector: " + str(output_vector))
                input_vector = output_vector

            self.y = self.output_activation.forward( np.dot( input_vector, self.Wy ) + self.by )
            print("Output layer vector: " + str(self.y))

        return self.y
        
    def __str__(self):
        info = "ANN(inputs: " + str(self.num_input_nodes) + ", outputs: " + str(self.num_output_nodes) + ", hidden layers: " + str(len(self.hidden_layers)) + ")\n"
        for i in range( len(self.hidden_layers) ):
            info += " Hidden layer " + str(i+1) + ": " + str(self.hidden_layers[i])
        info += "Weight Matrix: \n" + str(self.Wy) + "\n"
        info += "Biases Vector: \n" + str(self.by) + "\n"
        info += "Output activation function: " + str(self.output_activation)
        return info
