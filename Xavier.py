import numpy as np

def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return np.random.normal(size = [in_dim, out_dim], scale=xavier_stddev)

def initialize_NN(layers):  # layers = [2]+4*[50]+[6]
    weights = []
    biases = []
    num_layers = len(layers)
    for l in range(0,num_layers-1):
        W = xavier_init(size=[layers[l], layers[l+1]])
        b = np.zeros([1,layers[l+1]]) 
        weights.append(W)
        biases.append(b)
    return weights, biases

