import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image
import base64
import io

# Load pretrained model
model = tf.keras.models.load_model('mnist_model.h5')

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_data = data["image"]
    # Decode the base64 image
    image_data = base64.b64decode(image_data.split(",")[1])
    image = Image.open(io.BytesIO(image_data)).convert("L")
    image = image.resize((28, 28))  # Resize to MNIST size
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize
    image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for the model

    prediction = model.predict(image_array).argmax()
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
