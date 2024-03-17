import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

def plot_training(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

modelRNN = tf.keras.Sequential()
modelRNN.add(layers.SimpleRNN(64, input_shape=(28, 28)))
# add a RNN layer with 64 internal units / neurons for output
modelRNN.add(layers.Dense(10, activation='softmax'))
modelRNN.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
modelRNN.summary()
historyRNN = modelRNN.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
plot_training(historyRNN)
modelRNN.evaluate(x_test, y_test, verbose=2)

modelLSTM = tf.keras.Sequential()
modelLSTM.add(layers.LSTM(128, input_shape=(28, 28)))
# add a LSTM layer with 128 internal units / neurons for output
modelLSTM.add(layers.Dense(10, activation='softmax'))
modelLSTM.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
modelLSTM.summary()
historyLSTM = modelLSTM.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
plot_training(historyLSTM)
modelLSTM.evaluate(x_test, y_test, verbose=2)