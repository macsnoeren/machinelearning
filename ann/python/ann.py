import numpy as np
import matplotlib as plt

# Author: Maurice Snoeren - Avans Hogeschool - mac.snoeren@avans.nl
# 09-2020 - Beta version

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
        self.zy                = [] # Hold the summation of the input and bias with the weights
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

    def get_total_hidden_layers(self):
        return len(self.hidden_layers)

    def get_hidden_layer(self, i):
        if ( i < len(self.hidden_layers)):
            return self.hidden_layers[i]

        else:
            print("get_hidden_layer: error no layer found!")
            return -1

    def forward_propagation(self, x):
        """Calculate the output vector y of the neural network based on the x and hidden layers."""
        self.x = x
        print("Forward propagation\nInput layer vector: " + str(x))

        if ( len(self.hidden_layers) == 0):
            self.zy = np.dot( self.x, self.Wy ) + self.by
            self.y = self.output_activation.forward( self.zy )

        else: # Loop over the hidden layers
            input_vector = self.x
            for hidden_layer in self.hidden_layers:
                output_vector = hidden_layer.forward_propagation(input_vector)
                print("Hidden layer output vector: " + str(output_vector))
                input_vector = output_vector

            self.zy = np.dot( input_vector, self.Wy ) + self.by
            self.y  = self.output_activation.forward( self.zy )
            print("Output layer vector: " + str(self.y))

        return self.y
        
    def back_propagation(self, input_example, output_desired):
        """Perform calculation the network backwards"""

        self.forward_propagation(input_example)  # Perform first the forward propagation calculation

        J = 0.5 * ( self.y - output_desired )**2 # Calculate the cost
        print("Backward propagation\nJ = " + str(J))

        dJ_dy = ( self.y - output_desired )
        print("dJ_dy   : " + str(dJ_dy))
        dy_dzy   = self.output_activation.derivative( self.zy )
        print("dy_dzy  : " + str(dy_dzy))
        delta = np.multiply( dJ_dy, dy_dzy ) # required to back propagate through the network (part of the derivation that propagates back into the network)
        print("delta: " + str(delta))
        weights = self.Wy
        dJ_dWh = [] # Hold the weight gradients of the hidden layers

        if ( len(self.hidden_layers) == 0):
            dJ_dWy = np.dot( self.x.transpose(), delta )
            print("dJ_dWy  : " + str(dJ_dWy))

        else: # Loop over the hidden layers from back to start
            dzy_dWy = self.hidden_layers[ len(self.hidden_layers)-1 ].h
            print("dzy_dWy : " + str(dzy_dWy))
            dJ_dWy = np.dot( dzy_dWy.transpose(), delta )
            print("dJ_dWy  : " + str(dJ_dWy))

            dJ_dWh = []
            for hidden_layer in reversed(self.hidden_layers):
                result  = hidden_layer.back_propagation(delta, weights)
                delta   = result['delta']
                weights = result['W']
                dJ_dWh.append( result['dJ_dWh'])

        return {'dJ_dWh': list(reversed(dJ_dWh)), 'dJ_dWy': dJ_dWy }

    def __str__(self):
        info = "ANN(inputs: " + str(self.num_input_nodes) + ", outputs: " + str(self.num_output_nodes) + ", hidden layers: " + str(len(self.hidden_layers)) + ")\n"
        for i in range( len(self.hidden_layers) ):
            info += " Hidden layer " + str(i+1) + ": " + str(self.hidden_layers[i])
        info += "Weight Matrix: \n" + str(self.Wy) + "\n"
        info += "Biases Vector: \n" + str(self.by) + "\n"
        info += "Output activation function: " + str(self.output_activation)
        return info

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
        self.x                = [] # Input vector of this hidden layer
        self.Wh               = np.random.rand(num_input_nodes, num_hidden_nodes) # Hidden Weight Matrix
        self.bh               = np.random.rand(num_hidden_nodes)                  # Biases vector
        self.zh               = [] # Hold the summation of the input and bias with the weights
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
        self.x = x
        print("input vector (x): " + str(self.x))
        self.zh = np.dot(x, self.Wh) + self.bh
        print("zh: " + str(self.zh))
        self.h  = self.activation.forward( self.zh )
        return self.h

    def back_propagation(self, prev_delta, prev_W):
        """Part of the back propagation calculation of the ANN network"""
        print("Hidden Layer Back Prop")   
        # delta = dJ_dy * dy_dzy => zy output previous layer

        print("prev_delta: " + str(prev_delta))
        print("prev_W: " + str(prev_W))

        delta2 = np.dot( prev_delta, prev_W.transpose() ) * self.activation.derivative( self.zh )
        print("delta2: " + str(delta2))

        #dz1_dWh = self.h # This is not correct and it should be the activation from the previous layer
        dz1_dWh = self.x # Activation from the previous layer!
        print("dz1_dWh: " + str(dz1_dWh))
        dJ_dWh  = np.dot( dz1_dWh.transpose(), delta2 )
        print("dJ_dWh  : " + str(dJ_dWh))

        return { 'delta': delta2, 'W': self.Wh, 'dJ_dWh': dJ_dWh }

    def __str__(self):
        info = "(inputs: " + str(self.num_input_nodes) + ", nodes: " + str(self.num_hidden_nodes) + ", activation: " + str(self.activation) + ")\n"
        info += "  Weight Matrix: \n" + str(self.Wh) + "\n"
        info += "  Biases Vector: \n" + str(self.bh) + "\n"
        return info
