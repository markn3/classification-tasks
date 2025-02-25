import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt


def main():
    # load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # normalize pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = Sequential([
        Flatten(input_shape=(28,28)),
        Dense(128, activation='relu'),
        Dense(10, activation = 'softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test,y_test))

    model.save('mnist_model.keras')
    print("Model saved to mnist_model.keras")
if __name__ =="__main__":
    main()