import numpy as np

import copy

class NN():
    """
    Class encapsulating all the needed parameters for the neural network.
    Defined to always have only 3 layers - input, hidden, output.
    The size of these layers is modifiable and set to zero by default.
    There is no backpropagation implemented, only forward propagation.
    Weights are 2d arrays.

    input_layer_len : int
    hidden_layer_len : int
    output_layer_len : int
    """
    def __init__(self, input_layer_len=0, hidden_layer_len=0, output_layer_len=0):
        self.weights_ih = np.random.randn(input_layer_len, hidden_layer_len)
        self.weights_ho = np.random.randn(hidden_layer_len, output_layer_len)
    
    def from_weights(self, weights_ih, weights_ho):
        """
        Used to inicialize the neural network from other weights,
        usually already optimized.
        Creates deep copies of the weights given in the parameters and
        rewrites it's own ones with them.
        """
        self.weights_ih = copy.deepcopy(weights_ih)
        self.weights_ho = copy.deepcopy(weights_ho)
    
    def sigmoid(self, x):
        """
        Activation function used to make the computation
        non-linear.
        """
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, z):
        """
        Activation function to highlight the bigger numbers in the final
        output and lower even more numbers that are already low.
        This is not mine function, I did not write this, copied from 
        StackOverflow:
        https://stackoverflow.com/a/39558290
        """
        assert len(z.shape) == 2
        s = np.max(z, axis=1)
        s = s[:, np.newaxis] # necessary step to do broadcasting
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis] # dito
        return e_x / div
    
    def copy_weights(self):
        """
        Creates deepcopy of it's weights and returns them.
        """
        return copy.deepcopy(self.weights_ih), copy.deepcopy(self.weights_ho)

    def forward_propagate(self, x):
        """
        Main function of the neural network, used to predict output
        from the given input.
        It matrix multiplicate the input with the weights between two layers
        and then applies activation function on the result.
        """
        Zh = np.dot(x, self.weights_ih)
        Ah = self.sigmoid(Zh)
        Zo = np.dot(Ah, self.weights_ho)
        output = self.softmax(Zo)
        return output