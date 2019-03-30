import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    """
    Derivative of the sigmoid function
    """
    return sigmoid(x) * (1 - sigmoid(x))


learnrate = 0.5
x = np.array([1, 2, 3, 4])
y = np.array(0.5)

# Initial weights
w = np.array([0.5, -0.5, 0.3, 0.1])

# Calculate one gradient descent step for each weight
# Note: Some steps have been consolidated, so there are
# fewer variable names than in the above sample code

# TODO: Calculate the node’s linear combination of inputs and weights
h = x * w

# TODO: Calculate output of neural network
nn_output = sigmoid(h)

# TODO: Calculate error of neural network
error = np.sum(nn_output - y)

# TODO: Calculate the error term
# Remember, this requires the output gradient, which we haven’t
# specifically added a variable for.
error_term = sigmoid_prime(error)

# TODO: Calculate change in weights
del_w = error_term * learnrate

print('Neural Network output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)
