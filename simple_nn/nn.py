import numpy as np


def softmax_cross_entropy_for_logit(logits, y):
    true_logits = logits[range(len(logits)), y]
    return -true_logits + np.log(np.sum(np.exp(logits), axis=-1))


def softmax_cross_entropy_grad(logits, y):
    one_hot_encoding = np.zeros_like(logits)
    one_hot_encoding[range(len(logits)), y] = 1

    # compute every logits raised to the power of e
    exp = np.exp(logits)
    # compute normalizer along axis 1. the resulting shape will be (batch_size, 1)
    normalizer = np.sum(exp, axis=-1, keepdims=True)
    softmax = exp / normalizer
    return (-one_hot_encoding + softmax) / logits.shape[0]


def generate_batch(inputs, labels, batch_size):
    assert len(inputs) == len(labels)
    indices = np.random.permutation(len(inputs))
    for start_idx in range(0, len(inputs), batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield inputs[excerpt], labels[excerpt]


class TrainingHistory:
    loss = []
    train_accuracy = []
    val_accuracy = []


class SimpleNn:
    def __init__(self):
        self.layers = []
        self.history = None

    def add(self, layer):
        self.layers.append(layer)

    def compute_activations(self, train_x):
        activations = []
        left_input = train_x
        for layer in self.layers:
            activations.append(layer.forward(left_input))
            left_input = activations[-1]

        return [train_x] + activations

    def back_propagate(self, cross_entropy_loss_grad, activations):
        output_gradient = cross_entropy_loss_grad
        for i in range(len(self.layers), 0, -1):
            layer = self.layers[i - 1]
            input_gradient = layer.backward(activations[i - 1], output_gradient)
            output_gradient = input_gradient

    def fit(self, train_x, train_y, val_x, val_y, epochs, batch_size=32, keep_history=True, callbacks=[]):
        if keep_history:
            self.history = TrainingHistory()
        for epoch in range(epochs):
            batch_losses = []
            for x, y in generate_batch(train_x, train_y, batch_size):
                batch_losses.append(self.train_batch(x, y))

            train_accuracy = np.mean(self.predict(train_x) == train_y)
            val_accuracy = np.mean(self.predict(val_x) == val_y)
            loss = np.mean(batch_losses)
            if keep_history:
                self.history.loss.append(loss)
                self.history.train_accuracy.append(train_accuracy)
                self.history.val_accuracy.append(val_accuracy)

            if callbacks is not None:
                for callback in callbacks:
                    callback(epoch, train_accuracy, val_accuracy, loss)

    def predict(self, x):
        logits = self.compute_activations(x)[-1]
        return np.argmax(logits, axis=1)

    def train_batch(self, train_x, labels):
        activations = self.compute_activations(train_x)
        logits = activations[-1]
        cross_entropy_loss = softmax_cross_entropy_for_logit(logits, labels)
        cross_entropy_loss_grad = softmax_cross_entropy_grad(logits, labels)
        self.back_propagate(cross_entropy_loss_grad, activations)

        return np.mean(cross_entropy_loss)
