import os
import time
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL1_PATH = BASE_DIR / "model1_basic_cnn.keras"
MODEL2_PATH = BASE_DIR / "model2_improved_cnn.keras"

VALIDATION_DIR = BASE_DIR / "cats_and_dogs" / "validation"

IMG_SIZE = (150, 150)
BATCH_SIZE = 32


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("=" * 70)
print("LOADING VALIDATION DATA")
print("=" * 70)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = validation_dataset.class_names

print("Classes:", class_names)


# ============================================================
# NORMALISE PIXELS
# ============================================================

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

validation_dataset = validation_dataset.map(
    lambda images, labels:
        (normalization_layer(images), labels)
)

validation_dataset = validation_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def evaluate_model(model_name, model_path):

    print("\n" + "=" * 70)
    print(f"EVALUATING {model_name}")
    print("=" * 70)

    # Load model
    model = tf.keras.models.load_model(model_path)

    print("Model loaded:", model_path)
    print("Parameters:", f"{model.count_params():,}")

    # --------------------------------------------------------
    # Evaluate accuracy and loss
    # --------------------------------------------------------

    loss, accuracy = model.evaluate(
        validation_dataset,
        verbose=0
    )

    # --------------------------------------------------------
    # Measure inference time
    # --------------------------------------------------------

    start_time = time.perf_counter()

    predictions = model.predict(
        validation_dataset,
        verbose=0
    )

    end_time = time.perf_counter()

    inference_time = end_time - start_time

    # --------------------------------------------------------
    # Collect true labels
    # --------------------------------------------------------

    true_labels = []

    for _, labels in validation_dataset:
        true_labels.extend(
            labels.numpy().flatten().astype(int)
        )

    true_labels = np.array(true_labels)

    # --------------------------------------------------------
    # Convert predictions to labels
    # --------------------------------------------------------

    probabilities = predictions.flatten()

    predicted_labels = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Confusion matrix values
    # --------------------------------------------------------

    tp = int(
        np.sum(
            (true_labels == 1) &
            (predicted_labels == 1)
        )
    )

    tn = int(
        np.sum(
            (true_labels == 0) &
            (predicted_labels == 0)
        )
    )

    fp = int(
        np.sum(
            (true_labels == 0) &
            (predicted_labels == 1)
        )
    )

    fn = int(
        np.sum(
            (true_labels == 1) &
            (predicted_labels == 0)
        )
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Average confidence
    # --------------------------------------------------------

    confidence_scores = np.where(
        predicted_labels == 1,
        probabilities,
        1 - probabilities
    )

    average_confidence = float(
        np.mean(confidence_scores)
    )

    # --------------------------------------------------------
    # Model file size
    # --------------------------------------------------------

    size_mb = (
        os.path.getsize(model_path)
        / (1024 * 1024)
    )

    # --------------------------------------------------------
    # Result dictionary
    # --------------------------------------------------------

    result = {
        "model": model_name,
        "parameters": model.count_params(),
        "file_size_mb": size_mb,
        "validation_loss": float(loss),
        "validation_accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "average_confidence": average_confidence,
        "inference_time_seconds": inference_time,
        "images_evaluated": len(true_labels),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }

    return result


# ============================================================
# EVALUATE BOTH MODELS
# ============================================================

model1_results = evaluate_model(
    "Model 1 - Basic CNN",
    MODEL1_PATH
)

model2_results = evaluate_model(
    "Model 2 - Improved CNN",
    MODEL2_PATH
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 100)
print("MODEL COMPARISON")
print("=" * 100)

print(
    f"{'Metric':<30}"
    f"{'Model 1':>20}"
    f"{'Model 2':>20}"
)

print("-" * 70)

comparison_rows = [
    (
        "Parameters",
        f"{model1_results['parameters']:,}",
        f"{model2_results['parameters']:,}",
    ),
    (
        "File size (MB)",
        f"{model1_results['file_size_mb']:.2f}",
        f"{model2_results['file_size_mb']:.2f}",
    ),
    (
        "Validation accuracy",
        f"{model1_results['validation_accuracy']:.4f}",
        f"{model2_results['validation_accuracy']:.4f}",
    ),
    (
        "Validation loss",
        f"{model1_results['validation_loss']:.4f}",
        f"{model2_results['validation_loss']:.4f}",
    ),
    (
        "Precision",
        f"{model1_results['precision']:.4f}",
        f"{model2_results['precision']:.4f}",
    ),
    (
        "Recall",
        f"{model1_results['recall']:.4f}",
        f"{model2_results['recall']:.4f}",
    ),
    (
        "F1-score",
        f"{model1_results['f1_score']:.4f}",
        f"{model2_results['f1_score']:.4f}",
    ),
    (
        "Average confidence",
        f"{model1_results['average_confidence']:.4f}",
        f"{model2_results['average_confidence']:.4f}",
    ),
    (
        "Inference time (s)",
        f"{model1_results['inference_time_seconds']:.4f}",
        f"{model2_results['inference_time_seconds']:.4f}",
    ),
]

for metric, value1, value2 in comparison_rows:
    print(
        f"{metric:<30}"
        f"{value1:>20}"
        f"{value2:>20}"
    )


# ============================================================
# CONFUSION MATRICES
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX VALUES")
print("=" * 70)

print("\nModel 1:")
print(
    f"TN={model1_results['true_negative']}  "
    f"FP={model1_results['false_positive']}"
)

print(
    f"FN={model1_results['false_negative']}  "
    f"TP={model1_results['true_positive']}"
)

print("\nModel 2:")
print(
    f"TN={model2_results['true_negative']}  "
    f"FP={model2_results['false_positive']}"
)

print(
    f"FN={model2_results['false_negative']}  "
    f"TP={model2_results['true_positive']}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = Path(
    os.environ.get(
        "EVALUATION_OUTPUT",
        str(BASE_DIR / "model_evaluation_results.json")
    )
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            "model1": model1_results,
            "model2": model2_results,
        },
        file,
        indent=4
    )

print(
    "\nEvaluation results saved to:",
    output_path
)

print("\nEvaluation complete.")