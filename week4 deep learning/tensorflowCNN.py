import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers, losses, metrics, datasets


def plot_training(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()


(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.reshape(60000, 28, 28, 1)

model = tf.keras.Sequential([
    layers.Conv2D(32, (5, 5), padding="SAME", activation="relu"),
    # layers.Conv2D(1, (5, 5), padding="SAME", activation="relu"),
    # here we define the first convolution layer with 32 kernels/neurons of kernel size 5x5, activation function of relu
    layers.MaxPool2D(2, 2, padding="same"),
    # we define a max pooling with the size of 2x2, the input the the returned value after convolution
    layers.Conv2D(64, (5, 5), padding="SAME", activation="relu"),
    layers.MaxPool2D(2, 2, padding="same"),
    layers.Flatten(),
    layers.Dense(1024, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])


model.build((10000, 28, 28, 1))
model.summary()
# to review the structure of your network

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(x_train, y_train, epochs=5,
                    validation_data=(x_test, y_test))
plot_training(history)
model.evaluate(x_test, y_test, verbose=2)
# you are likely to have a similar accuracy in comparison with using NN only
