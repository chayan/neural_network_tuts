import simple_nn.nn as nn
import keras
from simple_nn.layers import Dense, ReLu


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype(float) / 255.
    x_test = x_test.astype(float) / 255.

    x_train = x_train.reshape([x_train.shape[0], -1])
    x_test = x_test.reshape([x_test.shape[0], -1])

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = load_data()

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

network = nn.SimpleNn()
network.add(Dense(x_train.shape[1], 150))
network.add(ReLu())
network.add(Dense(150, 250))
network.add(ReLu())
network.add(Dense(250, 10))


def callback(epoch, train_accuracy, val_accuracy, loss):
    print("Epoch %d" % epoch)
    print("Train accuracy: %.2f, Validation accuracy: %.2f, Loss: %.2f" % (train_accuracy, val_accuracy, loss))


network.fit(x_train, y_train, x_test, y_test, 20, 32, True, [callback])