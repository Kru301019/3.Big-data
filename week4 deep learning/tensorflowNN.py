# The MNIST dataset is a benchmark used in almost all languages
# You can load it from either TensorFlow lib or the keras lib
# this one uses softmax + NN + gradient descent /  + cross entropy to recognise MNIST
# CNN is not used here

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers
# from keras.datasets import mnist and convert pixel values to float between 0-1
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def plot_digit(index):
    digit = x_train[index]
    digit = np.array(digit, dtype='float')
    pixels = digit.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()
    print(y_train)
for i in range(5):
    plot_digit(i)

def plot_training(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()
# plot the training accuracy / validation accuracy changes

model = tf.keras.Sequential([
    layers.Flatten(input_shape=[28, 28]),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
# here we use the keras pipeline, you can use other ways to define the model
# or you can try another model with different layers and parameters
# model = models.Sequential([
#     layers.Flatten(input_shape=(28, 28)),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(30, activation='relu'),
#     layers.Dropout(0.2),
#     layers.Dense(10, activation='softmax')
# ])
model.summary()
# to review the structure of your network

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# adam is the commonly used optimiser today
# no convolution layer is used here, only the fully connected layers are used
# please do check what each layer represents and how they work together as a network

history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
# the accuracy is not always increasing if you follow the output during each epoch
# please do browse why this happens
plot_training(history)
model.evaluate(x_test, y_test, verbose=2)
# you are likely to have a training accuracy of 99.8x%-99.9x% and a validation accuracy rate of 98.x% for digits recognition