import os
import sys
import tensorflow as tf
from keras.utils import load_img, img_to_array


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model1_basic_cnn.keras"
)

IMG_SIZE = (150, 150)

CLASS_NAMES = ["cats", "dogs"]


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
# CHECK IMAGE ARGUMENT
# ============================================================

if len(sys.argv) < 2:
    print("\nUsage:")
    print("python predict.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]


if not os.path.exists(image_path):
    print("\nError: Image file not found.")
    print("Path:", image_path)
    sys.exit(1)


# ============================================================
# LOAD IMAGE
# ============================================================

image = load_img(
    image_path,
    target_size=IMG_SIZE
)

image_array = img_to_array(image)


# ============================================================
# NORMALISE IMAGE
# ============================================================

image_array = image_array / 255.0


# ============================================================
# ADD BATCH DIMENSION
# ============================================================

image_array = tf.expand_dims(
    image_array,
    axis=0
)


# ============================================================
# MAKE PREDICTION
# ============================================================

prediction = model.predict(
    image_array,
    verbose=0
)[0][0]


# ============================================================
# INTERPRET PREDICTION
# ============================================================

if prediction >= 0.5:
    predicted_class = "dogs"
    confidence = prediction
else:
    predicted_class = "cats"
    confidence = 1 - prediction


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print("Image:", os.path.basename(image_path))
print("Prediction:", predicted_class.upper())
print(f"Confidence: {confidence * 100:.2f}%")

print("=" * 60)