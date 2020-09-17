import numpy as np
from ann import ANN, ANN_Sigmoid_Activation, ANN_Hidden_Layer

sa  = ANN_Sigmoid_Activation()
ann = ANN(4, 2, sa)
ann.add_hidden_layer(ANN_Hidden_Layer(4, 5, sa))
ann.add_hidden_layer(ANN_Hidden_Layer(5, 5, sa))
#ann.add_hidden_layer(ANN_Hidden_Layer(10, 10, sa))
#ann.add_hidden_layer(ANN_Hidden_Layer(10, 10, sa))
#ann.add_hidden_layer(ANN_Hidden_Layer(10, 10, sa))
#ann.set_biases_vector([[0, 0]])
#ann.set_weight_matrix([[1, 1],
#                       [1, 1],
#                       [1, 1],
#                       [1, 1]])
print(ann)

x = np.array([[0.1, 0.1, 0.1, 0.1]])

print( ann.forward_propagation(x) )

x = np.array([[1, 1, 1, 1]])
y = np.array([[1, 0]])
result = ann.back_propagation(x, y)

print("results\n\n-------\n\n" + str(result))