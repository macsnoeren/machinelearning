from ann import ANN, ANN_Sigmoid_Activation, ANN_Hidden_Layer

sa  = ANN_Sigmoid_Activation()
ann = ANN(4, 2, sa)
ann.add_hidden_layer(ANN_Hidden_Layer(4, 10, sa))
ann.add_hidden_layer(ANN_Hidden_Layer(10, 10, sa))
ann.add_hidden_layer(ANN_Hidden_Layer(10, 10, sa))
ann.add_hidden_layer(ANN_Hidden_Layer(10, 10, sa))

print(ann)