# ============================================================
# KERAS SENTIMENT ANALYSIS MODEL EVALUATION
# IMDB Reviews + GloVe + LSTM
# ============================================================

import os
import time

import numpy as np
import keras

from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences


# ============================================================
# 1. CONFIGURATION
# ============================================================

print("=" * 75)
print("KERAS SENTIMENT ANALYSIS MODEL EVALUATION")
print("=" * 75)

print("Keras version:", keras.__version__)
print("Backend:", keras.backend.backend())


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "keras_sentiment_model.keras"
)

VOCAB_SIZE = 10000
MAX_LENGTH = 200


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

print("\n" + "=" * 75)
print("LOADING TRAINED MODEL")
print("=" * 75)


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


model = keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")
print("Model:", MODEL_PATH)


# ============================================================
# 3. LOAD TEST DATA
# ============================================================

print("\n" + "=" * 75)
print("LOADING IMDB TEST DATA")
print("=" * 75)


(
    _
    ,
    _
), (
    x_test,
    y_test
) = imdb.load_data(
    num_words=VOCAB_SIZE
)


x_test = pad_sequences(
    x_test,
    maxlen=MAX_LENGTH,
    padding="pre",
    truncating="pre"
)


print(
    "Test reviews:",
    len(x_test)
)

print(
    "Test shape:",
    x_test.shape
)


# ============================================================
# 4. MODEL PARAMETER COUNT
# ============================================================

total_parameters = model.count_params()

trainable_parameters = sum(
    np.prod(variable.shape)
    for variable
    in model.trainable_weights
)

non_trainable_parameters = sum(
    np.prod(variable.shape)
    for variable
    in model.non_trainable_weights
)


# ============================================================
# 5. MODEL FILE SIZE
# ============================================================

model_size_mb = (
    os.path.getsize(MODEL_PATH)
    / (1024 * 1024)
)


# ============================================================
# 6. TEST-SET LOSS + ACCURACY
# ============================================================

print("\n" + "=" * 75)
print("TEST-SET EVALUATION")
print("=" * 75)


test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)


print(
    "Test loss:",
    f"{test_loss:.4f}"
)

print(
    "Test accuracy:",
    f"{test_accuracy:.4f}"
)


# ============================================================
# 7. GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 75)
print("GENERATING TEST PREDICTIONS")
print("=" * 75)


probabilities = model.predict(
    x_test,
    batch_size=128,
    verbose=1
).reshape(-1)


predictions = (
    probabilities >= 0.5
).astype(int)


# ============================================================
# 8. CONFUSION MATRIX VALUES
# ============================================================

true_positive = int(
    np.sum(
        (predictions == 1)
        & (y_test == 1)
    )
)

true_negative = int(
    np.sum(
        (predictions == 0)
        & (y_test == 0)
    )
)

false_positive = int(
    np.sum(
        (predictions == 1)
        & (y_test == 0)
    )
)

false_negative = int(
    np.sum(
        (predictions == 0)
        & (y_test == 1)
    )
)


# ============================================================
# 9. PRECISION, RECALL, F1
# ============================================================

precision = (
    true_positive
    / (
        true_positive
        + false_positive
    )
    if (
        true_positive
        + false_positive
    ) > 0
    else 0.0
)


recall = (
    true_positive
    / (
        true_positive
        + false_negative
    )
    if (
        true_positive
        + false_negative
    ) > 0
    else 0.0
)


f1_score = (
    2
    * precision
    * recall
    / (
        precision
        + recall
    )
    if (
        precision
        + recall
    ) > 0
    else 0.0
)


# ============================================================
# 10. CONFIDENCE
# ============================================================

confidence_scores = np.where(
    predictions == 1,
    probabilities,
    1 - probabilities
)

average_confidence = float(
    np.mean(
        confidence_scores
    )
)


# ============================================================
# 11. INFERENCE LATENCY
# ============================================================

print("\n" + "=" * 75)
print("INFERENCE LATENCY")
print("=" * 75)


sample = x_test[
    0:1
]


# Warm-up
for _ in range(10):

    model.predict(
        sample,
        verbose=0
    )


NUMBER_OF_RUNS = 100


start_time = time.perf_counter()


for _ in range(
    NUMBER_OF_RUNS
):

    model.predict(
        sample,
        verbose=0
    )


end_time = time.perf_counter()


average_latency_ms = (
    (
        end_time
        - start_time
    )
    / NUMBER_OF_RUNS
    * 1000
)


print(
    "Measured runs:",
    NUMBER_OF_RUNS
)

print(
    "Average inference latency:",
    f"{average_latency_ms:.3f} ms"
)


# ============================================================
# 12. LOAD WORD INDEX FOR CUSTOM TEXT
# ============================================================

word_index = imdb.get_word_index()


adjusted_word_index = {
    word: index + 3
    for word, index
    in word_index.items()
}


adjusted_word_index["<pad>"] = 0
adjusted_word_index["<start>"] = 1
adjusted_word_index["<unk>"] = 2
adjusted_word_index["<unused>"] = 3


# ============================================================
# 13. CUSTOM REVIEW PREPROCESSING
# ============================================================

def preprocess_review(review):

    review = (
        review
        .lower()
        .strip()
    )


    punctuation = (
        '.,!?;:"()[]{}'
    )


    for symbol in punctuation:

        review = review.replace(
            symbol,
            " "
        )


    words = review.split()


    sequence = [
        1
    ]


    for word in words:

        word_id = adjusted_word_index.get(
            word,
            2
        )


        if word_id >= VOCAB_SIZE:

            word_id = 2


        sequence.append(
            word_id
        )


    sequence = pad_sequences(
        [sequence],
        maxlen=MAX_LENGTH,
        padding="pre",
        truncating="pre"
    )


    return sequence


# ============================================================
# 14. CUSTOM REVIEW PREDICTION
# ============================================================

def predict_review(review):

    processed = preprocess_review(
        review
    )


    probability = float(
        model.predict(
            processed,
            verbose=0
        )[0][0]
    )


    if probability >= 0.5:

        label = "POSITIVE"
        confidence = probability

    else:

        label = "NEGATIVE"
        confidence = (
            1 - probability
        )


    return (
        label,
        confidence,
        probability
    )


# ============================================================
# 15. QUALITATIVE EXAMPLES
# ============================================================

print("\n" + "=" * 75)
print("QUALITATIVE SENTIMENT DEMONSTRATION")
print("=" * 75)


demo_reviews = [

    "This movie was fantastic and I absolutely loved it.",

    "The acting was excellent and the story was brilliant.",

    "This was a terrible movie and a complete waste of time.",

    "The film was boring disappointing and badly made.",

    "The movie was okay but nothing special."

]


for review in demo_reviews:

    label, confidence, probability = (
        predict_review(
            review
        )
    )


    print(
        "\nReview:",
        review
    )

    print(
        "Prediction:",
        label
    )

    print(
        "Confidence:",
        f"{confidence * 100:.2f}%"
    )

    print(
        "Raw sigmoid score:",
        f"{probability:.4f}"
    )


# ============================================================
# 16. FINAL EVALUATION SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL MODEL EVALUATION")
print("=" * 75)


print(
    f"{'Metric':<38}"
    f"{'Result':>20}"
)

print("-" * 58)


print(
    f"{'Test reviews':<38}"
    f"{len(x_test):>20,}"
)


print(
    f"{'Test accuracy':<38}"
    f"{test_accuracy * 100:>19.2f}%"
)


print(
    f"{'Test loss':<38}"
    f"{test_loss:>20.4f}"
)


print(
    f"{'Precision':<38}"
    f"{precision * 100:>19.2f}%"
)


print(
    f"{'Recall':<38}"
    f"{recall * 100:>19.2f}%"
)


print(
    f"{'F1-score':<38}"
    f"{f1_score * 100:>19.2f}%"
)


print(
    f"{'Average confidence':<38}"
    f"{average_confidence * 100:>19.2f}%"
)


print(
    f"{'True positives':<38}"
    f"{true_positive:>20,}"
)


print(
    f"{'True negatives':<38}"
    f"{true_negative:>20,}"
)


print(
    f"{'False positives':<38}"
    f"{false_positive:>20,}"
)


print(
    f"{'False negatives':<38}"
    f"{false_negative:>20,}"
)


print(
    f"{'Total parameters':<38}"
    f"{total_parameters:>20,}"
)


print(
    f"{'Trainable parameters':<38}"
    f"{int(trainable_parameters):>20,}"
)


print(
    f"{'Non-trainable parameters':<38}"
    f"{int(non_trainable_parameters):>20,}"
)


print(
    f"{'Model file size':<38}"
    f"{model_size_mb:>17.3f} MB"
)


print(
    f"{'Average inference latency':<38}"
    f"{average_latency_ms:>17.3f} ms"
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)


print()

print(
    "                    Predicted"
)

print(
    "                 Negative  Positive"
)

print(
    f"Actual Negative   "
    f"{true_negative:8d}"
    f"  "
    f"{false_positive:8d}"
)

print(
    f"Actual Positive   "
    f"{false_negative:8d}"
    f"  "
    f"{true_positive:8d}"
)


# ============================================================
# 18. INTERPRETATION
# ============================================================

print("\n" + "=" * 75)
print("INTERPRETATION")
print("=" * 75)


print(
    "Unlike the PyTorch demonstration, these metrics are "
    "calculated on 25,000 held-out IMDB test reviews."
)

print(
    "Therefore, the accuracy, precision, recall and F1-score "
    "measure generalisation to unseen data rather than "
    "memorisation of the training examples."
)


print("\nEvaluation complete.")