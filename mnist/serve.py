from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load the model (make sure mnist_model.h5 is in the same directory)
model = tf.keras.models.load_model("mnist_model.keras")

def prepare_image(image, target_size=(28, 28)):
    # Convert to grayscale if not already
    if image.mode != "L":
        image = image.convert("L")
    # Resize to target size
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  # Normalize
    image = image.reshape(1, 28, 28)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    image_data = data["image"]
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        image = prepare_image(image)
        prediction = model.predict(image)
        digit = int(np.argmax(prediction[0]))
        return jsonify({"digit": digit})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run on all interfaces so Docker can map the port
    app.run(host='0.0.0.0', port=5000)
