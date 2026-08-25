import os
import io

from flask import Flask, request, jsonify, render_template  # type: ignore

import tensorflow as tf
import numpy as np


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model1_basic_cnn.keras"
)

IMG_SIZE = (150, 150)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("=" * 60)
print("LOADING TRAINED MODEL")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print("Model:", MODEL_PATH)


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# HEALTH CHECK ROUTE
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "cats-dogs-ai",
        "model": "model1_basic_cnn"
    }), 200


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No image provided"
        }), 400

    image_file = request.files["image"]

    try:

        # ----------------------------------------------------
        # Read uploaded image into memory
        # ----------------------------------------------------

        image_bytes = image_file.read()

        # ----------------------------------------------------
        # Convert bytes into an image
        # ----------------------------------------------------

        image = tf.keras.utils.load_img(
            io.BytesIO(image_bytes),
            target_size=IMG_SIZE
        )

        # ----------------------------------------------------
        # Convert image to NumPy array
        # ----------------------------------------------------

        image_array = tf.keras.utils.img_to_array(image)

        # ----------------------------------------------------
        # Normalise pixel values
        # ----------------------------------------------------

        image_array = image_array / 255.0

        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        prediction = model.predict(
            image_array,
            verbose=0
        )[0][0]

        # ----------------------------------------------------
        # Interpret binary classification
        # ----------------------------------------------------

        if prediction >= 0.5:

            label = "DOGS"
            confidence = float(prediction)

        else:

            label = "CATS"
            confidence = float(1 - prediction)

        # ----------------------------------------------------
        # Return prediction
        # ----------------------------------------------------

        return jsonify({
            "prediction": label,
            "confidence": confidence
        }), 200

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("STARTING FLASK INFERENCE API")
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )