

'''
Goal: Implement Vanilla Neural Deep Network using only numpy

Requirements:
    -Be able to take different network architectures and compute gradients
    -Instantiate weights using special instantiation (He insantiation)
    -Use RELU activation for all hidden layers
    -Use Softmax for Output layer
    -Implement Regularization + Dropout
    -Implement Adam Optimizer for Gradient Descent + Minibatching
    -Normalize intermediate datasets
    -Test peformance on MNIST dataset


Checklist:
    -Implement network setup - DONE
    -Implement network instantiation - DONE
    -Implement forward prop - DONE
        -with relu - DONE
        -with softmax - DONE
        -with normalizatization
    -Implement cost function - DONE
    -Implement back prop - DONE
        -with minibatching and batch normalization
        -with Adam SGD
    -Implement train function to train network based on hyperparameters - DONE
'''

import numpy as np

class DeepNet(object):

    def __init__(self, **kwargs):
        self.layer_dims = kwargs.get('layer_dims') # e.g. [X, W1, W2, Y]

        self.param_cache = {} # Index: Layer -> Map: {W,b}

        # Instantiate Neural Netowrk parameters (W,b) for each hidden layer
        for i in range(1, len(self.layer_dims)):
            W,b = self.init_layer_param((self.layer_dims[i], self.layer_dims[i-1]))
            self.param_cache[i] = {}
            self.param_cache[i]['W'] = W
            self.param_cache[i]['b'] = b

    def init_layer_param(self, shape):
        W = np.random.randn(*shape) * np.sqrt(2/shape[1]) # He instantiation
        b = np.zeros((shape[0], 1))
        return (W,b)

    # Forward prop for minibatch
    def forwardprop(self, X):
        m = X.shape[1]

        A = X
        Y_hat = None
        for i in range(1, len(self.layer_dims)):

            # Cache the inputs from previous layer for backprop
            self.param_cache[i]['A_prev'] = A

            W = self.param_cache[i]['W']
            b = self.param_cache[i]['b']

            Z = np.dot(W,A) + b
            self.param_cache[i]['Z'] = Z

            # Last Layer, use SoftMax
            if i == len(self.layer_dims) - 1:
                Y_hat = self.softmax(Z)

            # Other layers, use RELU
            else:
                A = self.relu(Z)

        return Y_hat

    def relu(self, Z):
        A = np.maximum(0, Z)
        return A

    def softmax(self,Z):
        exp = np.exp(Z)
        sums = np.sum(exp, axis=0)
        Y_hat = np.divide(exp,sums)
        return Y_hat

    def crossentropy_loss(self, Y, Y_hat):
        m = Y.shape[1]
        log_Y_hat = np.log(Y_hat)
        loss = -1/m*np.sum(np.multiply(Y,log_Y_hat))
        return loss

    def relu_backward(self, X):
        return (X > 0)

    def backprop(self, Y, Y_hat):

        m = Y.shape[1]

        for i in range(len(self.layer_dims) - 1, 0, -1):

            W = self.param_cache[i]['W']
            b = self.param_cache[i]['b']
            Z = self.param_cache[i]['Z']
            A_prev = self.param_cache[i]['A_prev']

            # If last layer, use CrossEntropy Loss for dZ:
            if i == len(self.layer_dims) - 1:
                dZ = Y_hat - Y

            else:
                dZ = np.multiply(dA,self.relu_backward(Z))

            # Compute parameter gradients
            dW = 1/m * np.dot(dZ, A_prev.T)
            db = 1/m * np.sum(dZ, axis=1, keepdims = True)

            # Update parameters via gradient descent
            W = W - self.learning_rate * dW
            b = b - self.learning_rate * db

            # Store updated parameters
            self.param_cache[i]['W'] = W
            self.param_cache[i]['b'] = b

            # Compute output gradient of previous layer for next iteration
            dA = np.dot(W.T, dZ)


    def train(self, X, Y, iterations = 1500, **kwargs):
        self.minibatch = kwargs['minibatch']
        self.learning_rate = kwargs['learning_rate']

        for i in range(iterations):
            Y_hat = self.forwardprop(X)
            cost  = self.crossentropy_loss(Y, Y_hat)
            print("Iteration", i,":", cost)
            self.backprop(Y,Y_hat)

    def save(self, path):
        f = open(path, 'wb')
        f.dumps(self.param_cache)
        f.close()
