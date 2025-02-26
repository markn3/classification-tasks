import tensorflow as tf
from tensorflow.keras.models import load_model

def main():
    # Load the model (make sure mnist_model.h5 is in the same directory)
    model = tf.keras.models.load_model("mnist_model.keras")

if __name__ == "__main__":
    main()
