from NNModel.nn import DeepNet

import mnist
import numpy as np

# Download MNIST Data and Labels
train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Reshape Input Data for Images
X_Train = train_images.reshape(train_images.shape[0], train_images.shape[1]*train_images.shape[2]).T
X_Test  = test_images.reshape(test_images.shape[0], test_images.shape[1]*test_images.shape[2]).T
X_Train = X_Train/255 # Normalize
X_Test  = X_Test/255  # Normalize
print(X_Train.shape)
print(X_Test.shape)

# One hot encode data for training labels
train_labels = train_labels.reshape(1, train_labels.shape[0])
test_labels  = test_labels.reshape(1, test_labels.shape[0])
Y_Train = np.zeros((10, train_labels.shape[1]))
Y_Test  = np.zeros((10, test_labels.shape[1]))

Y_Train[train_labels, np.arange(Y_Train.shape[1])] = 1
Y_Test[test_labels, np.arange(Y_Test.shape[1])] = 1
print(Y_Train.shape)
print(Y_Test.shape)

# Set up network architecture. In this version we have 28*28 inputs, representing the image, two hidden layers of 32 and 32,
# and one output layer corresponding to softmax with 10 elements (10 digitsS)
kwargs = {'layer_dims': [28*28,32,32,10]}
m = 60000

net = DeepNet(**kwargs)
net.train(X_Train,Y_Train, learning_rate = 0.01, minibatch =3)

# Compute Accuracy
predictions_train = np.argmax(net.forwardprop(X_Train), axis = 0)
predictions_test = np.argmax(net.forwardprop(X_Test), axis = 0)
truths_train     = np.argmax(Y_Train, axis = 0)
truths_test      = np.argmax(Y_Test, axis = 0)

# Train
num_correct_train = np.sum(predictions_train == truths_train)
accuracy_train = num_correct_train/len(truths_train)*100.0
print("Train Accuracy: " , accuracy_train)

# Test
num_correct_test = np.sum(predictions_test== truths_test)
accuracy_test = num_correct_test/len(truths_test)*100.0
print("Test Accuracy: " , accuracy_test)
