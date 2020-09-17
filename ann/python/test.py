import numpy as np
from ann import ANN, ANN_Sigmoid_Activation, ANN_Hidden_Layer

sa  = ANN_Sigmoid_Activation()
ann = ANN(4, 2, sa)
ann.add_hidden_layer(ANN_Hidden_Layer(4, 10, sa))
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
alpha = 0.01

for i in range(100):
    result = ann.back_propagation(x, y)
    #print("results\n\n-------\n\n" + str(result))
    ann.set_weight_matrix(ann.get_weight_matrix() - alpha*result['dJ_dWy'])
    for i in range(ann.get_total_hidden_layers()):
        hl = ann.get_hidden_layer(i)
        wm = result['dJ_dWh'][i]
        hl.set_weight_matrix( hl.get_weight_matrix() - alpha*wm )



print( ann.forward_propagation(x) )