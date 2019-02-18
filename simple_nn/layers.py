import numpy as np


class Layer:
    """
    Base Layer class. Each layer is capable of performing two things:

    - Process input to get output:           output = layer.forward(left_input)

    - Propagate gradients through itself:    grad_input = layer.backward(left_input, output_gradient)
    """

    def __init__(self):
        # Base layer does nothing
        pass

    def forward(self, left_input):
        """
        Takes input from previous layer A [batch, input_units], returns layer output Z [batch, output_units]
        """
        # The base layer just returns whatever it gets as input.
        return left_input

    def backward(self, left_input, output_gradient):
        """
        Back-propagates through this layer.
        - Performs one step of gradient descent.
          Computes the gradient of Loss function w.r.t layer parameters (dW) and updates the params W as
          W -= eta * dW
        - Computes the gradient w.r.t the input (dA) and propagates to the previous layer

        Parameters
        -----------
        left_input:         input to this layer from previous layer A [batch, input_units]
        output_gradient:    gradient of Loss (J) w.r.t the output of this layer: dZ [batch, output_units].
                            During back propagation this is computed by the next layer propagated back to this layer

        Returns
        -----------
        input_gradient:     Loss gradient w.r.t the input to this layer which is propagated back to the previous layer
        """

        # for base layer there is no parameters to update and the input_gradient is same as output_gradient as dA/dZ = I

        return output_gradient


class ReLu(Layer):
    """
    Apply element wise Rectified linear unit
    ReLu(x) = max(x, 0)
    """
    def __init__(self):
        pass

    def forward(self, left_input):
        return np.maximum(left_input, 0)

    def backward(self, left_input, output_gradient):
        grad = left_input > 0
        return output_gradient * grad


class SoftMax(Layer):

    def __init__(self):
        pass

    def forward(self, left_input):
        # compute every logits raised to the power of e
        exp = np.exp(left_input)
        # compute normalizer along axis 1. the resulting shape will be (batch_size, 1)
        normalizer = np.sum(exp, axis=-1, keepdims=True)
        return exp/normalizer

    def backward(self, left_input, output_gradient):
        output = self.forward(left_input)
        grad = output * (1 - output)
        return output_gradient * grad


class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        """
        A layer where every input units are connected to every output unit with a dedicated link.
        Every link has a weight and is maintained in the weight parameters W [input_units, output_units]
        and bias parameters b [output_units]
        In forward pass, it performs an affine transform Z = A.W + b
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.weights = np.random.randn(input_units, output_units) * 0.01
        self.biases = np.zeros(output_units)

    def forward(self, left_input):
        return np.dot(left_input, self.weights) + self.biases

    def backward(self, left_input, output_gradient):
        input_gradient = np.dot(output_gradient, self.weights.T)

        # gradient w.r.t. weights and biases
        grad_weights = np.dot(left_input.T, output_gradient)
        grad_biases = np.sum(output_gradient, axis=0)

        # stochastic gradient descent step
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return input_gradient




