# ============================================================
# TensorFlow / Keras Cats-vs-Dogs Classification Demo
# Master Classroom Version
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# 2. CONFIGURATION
# ============================================================

print("TensorFlow version:", tf.__version__)

# Get the folder where this Python file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset locations
train_dir = os.path.join(
    BASE_DIR,
    "cats_and_dogs",
    "train"
)

validation_dir = os.path.join(
    BASE_DIR,
    "cats_and_dogs",
    "validation"
)

print("Training data:", train_dir)
print("Validation data:", validation_dir)

# Image and training configuration
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10


# ============================================================
# 3. LOAD THE DATASET
# ============================================================

print("\n" + "=" * 60)
print("LOADING DATASET")
print("=" * 60)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

# Class names
class_names = train_dataset.class_names

print("\nClasses:", class_names)


# ============================================================
# 4. VISUALISE TRAINING IMAGES
# ============================================================

print("\nDisplaying sample training images...")

plt.figure(figsize=(10, 10))

for images, labels in train_dataset.take(1):

    for i in range(9):

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(
            images[i].numpy().astype("uint8")
        )

        plt.title(
            class_names[int(labels[i][0])]
        )

        plt.axis("off")

plt.suptitle(
    "Sample Training Images",
    fontsize=16
)

plt.tight_layout()
plt.show()


# ============================================================
# 5. INSPECT IMAGE DATA
# ============================================================

for images, labels in train_dataset.take(1):

    print("\nImage shape:", images[0].shape)

    print("Example pixel values:")
    print(images[0].numpy()[0, 0])


# ============================================================
# 6. NORMALISE PIXEL VALUES
# ============================================================

print("\nNormalising pixel values...")

normalization_layer = tf.keras.layers.Rescaling(
    1.0 / 255
)

train_dataset = train_dataset.map(
    lambda images, labels:
        (normalization_layer(images), labels)
)

validation_dataset = validation_dataset.map(
    lambda images, labels:
        (normalization_layer(images), labels)
)

print("Pixel values have been normalised to [0, 1].")


# ============================================================
# 7. PREFETCH DATA
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# 8. MODEL 1 — BASIC CNN
# ============================================================

print("\n" + "=" * 60)
print("MODEL 1: BASIC CNN")
print("=" * 60)

model1 = tf.keras.Sequential([

    # Input
    tf.keras.Input(
        shape=(150, 150, 3)
    ),

    # Convolutional block 1
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Convolutional block 2
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Flatten feature maps
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Binary classification
    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# 9. COMPILE MODEL 1
# ============================================================

model1.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 10. DISPLAY MODEL 1
# ============================================================

model1.summary()


# ============================================================
# 11. TRAIN MODEL 1
# ============================================================

print("\nTraining Model 1...")

history1 = model1.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)


# ============================================================
# 12. MODEL 2 — IMPROVED CNN
# ============================================================

print("\n" + "=" * 60)
print("MODEL 2: IMPROVED CNN")
print("=" * 60)

model2 = tf.keras.Sequential([

    # Input
    tf.keras.Input(
        shape=(150, 150, 3)
    ),

    # --------------------------------------------------------
    # Data augmentation
    # --------------------------------------------------------

    tf.keras.layers.RandomFlip(
        "horizontal"
    ),

    tf.keras.layers.RandomRotation(
        0.05
    ),

    tf.keras.layers.RandomZoom(
        0.05
    ),

    # --------------------------------------------------------
    # Convolutional block 1
    # --------------------------------------------------------

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # Convolutional block 2
    # --------------------------------------------------------

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # Convolutional block 3
    # --------------------------------------------------------

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # Global average pooling
    # --------------------------------------------------------

    tf.keras.layers.GlobalAveragePooling2D(),

    # --------------------------------------------------------
    # Regularisation
    # --------------------------------------------------------

    tf.keras.layers.Dropout(
        0.3
    ),

    # --------------------------------------------------------
    # Classification layers
    # --------------------------------------------------------

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# 13. COMPILE MODEL 2
# ============================================================

model2.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 14. DISPLAY MODEL 2
# ============================================================

model2.summary()


# ============================================================
# 15. EARLY STOPPING
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# ============================================================
# 16. TRAIN MODEL 2
# ============================================================

print("\nTraining Model 2...")

history2 = model2.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)


# ============================================================
# 17. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

model1_train_acc = history1.history["accuracy"][-1]
model1_val_acc = history1.history["val_accuracy"][-1]

model2_train_acc = history2.history["accuracy"][-1]
model2_val_acc = history2.history["val_accuracy"][-1]

print(
    f"Model 1 parameters: "
    f"{model1.count_params():,}"
)

print(
    f"Model 2 parameters: "
    f"{model2.count_params():,}"
)

print(
    f"Model 1 final training accuracy: "
    f"{model1_train_acc:.4f}"
)

print(
    f"Model 1 final validation accuracy: "
    f"{model1_val_acc:.4f}"
)

print(
    f"Model 2 final training accuracy: "
    f"{model2_train_acc:.4f}"
)

print(
    f"Model 2 final validation accuracy: "
    f"{model2_val_acc:.4f}"
)


# ============================================================
# 18. PLOT MODEL 1 PERFORMANCE
# ============================================================

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)

plt.plot(
    history1.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history1.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Model 1: Training vs Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()


# Loss
plt.subplot(1, 2, 2)

plt.plot(
    history1.history["loss"],
    label="Training Loss"
)

plt.plot(
    history1.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Model 1: Training vs Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 19. PLOT MODEL 2 PERFORMANCE
# ============================================================

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)

plt.plot(
    history2.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history2.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Model 2: Training vs Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()


# Loss
plt.subplot(1, 2, 2)

plt.plot(
    history2.history["loss"],
    label="Training Loss"
)

plt.plot(
    history2.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Model 2: Training vs Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("DEMONSTRATION COMPLETE")
print("=" * 60)

print("\nKey concepts demonstrated:")
print("1. Loading image data")
print("2. Image visualisation")
print("3. Pixel normalisation")
print("4. CNN architecture")
print("5. Convolution and pooling")
print("6. Binary classification")
print("7. Data augmentation")
print("8. Dropout regularisation")
print("9. Early stopping")
print("10. Training vs validation performance")
print("11. Overfitting and generalisation")

# ============================================================
# TEST SET: LOAD UNSEEN IMAGES
# ============================================================

test_dir = os.path.join(
    BASE_DIR,
    "cats_and_dogs",
    "test"
)

print("\n" + "=" * 60)
print("TEST SET")
print("=" * 60)

test_images = []

for filename in os.listdir(test_dir):
    file_path = os.path.join(test_dir, filename)

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        test_images.append(file_path)

print("Number of test images:", len(test_images))


# ============================================================
# SINGLE IMAGE PREDICTION
# ============================================================
from keras.utils import load_img, img_to_array
image_path = test_images[0]

image = load_img(
    image_path,
    target_size=IMG_SIZE
)

image_array = img_to_array(image)

# Normalise pixel values
image_array = image_array / 255.0

# Add batch dimension
image_array = tf.expand_dims(
    image_array,
    axis=0
)

prediction = model1.predict(image_array, verbose=0)[0][0]

if prediction >= 0.5:
    predicted_class = "dogs"
else:
    predicted_class = "cats"

print("\nTest image:", os.path.basename(image_path))
print("Prediction:", predicted_class)
print("Probability:", float(prediction))

# ============================================================
# SAVE TRAINED MODELS
# ============================================================

print("\n" + "=" * 60)
print("SAVING MODELS")
print("=" * 60)

model1_path = os.path.join(BASE_DIR, "model1_basic_cnn.keras")
model2_path = os.path.join(BASE_DIR, "model2_improved_cnn.keras")

model1.save(model1_path)
model2.save(model2_path)

print("Model 1 saved to:", model1_path)
print("Model 2 saved to:", model2_path)

print("\nModels saved successfully.")
