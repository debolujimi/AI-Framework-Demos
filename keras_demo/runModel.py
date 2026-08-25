# ============================================================
# KERAS SENTIMENT ANALYSIS INFERENCE
# Loads the trained model and predicts review sentiment
# ============================================================

import os

import numpy as np
import keras

from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences


# ============================================================
# 1. CONFIGURATION
# ============================================================

print("=" * 70)
print("KERAS SENTIMENT ANALYSIS INFERENCE")
print("=" * 70)

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

print("\n" + "=" * 70)
print("LOADING TRAINED MODEL")
print("=" * 70)


if not os.path.exists(
    MODEL_PATH
):

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


model = keras.models.load_model(
    MODEL_PATH
)


print(
    "Model loaded successfully."
)

print(
    "Model:",
    MODEL_PATH
)


# ============================================================
# 3. LOAD IMDB WORD INDEX
# ============================================================

print("\n" + "=" * 70)
print("LOADING IMDB WORD INDEX")
print("=" * 70)


word_index = imdb.get_word_index()


# IMDB reserves:
# 0 = <pad>
# 1 = <start>
# 2 = <unk>
# 3 = <unused>
#
# Therefore actual words are shifted by 3.

adjusted_word_index = {

    word: index + 3

    for word, index
    in word_index.items()

}


adjusted_word_index[
    "<pad>"
] = 0


adjusted_word_index[
    "<start>"
] = 1


adjusted_word_index[
    "<unk>"
] = 2


adjusted_word_index[
    "<unused>"
] = 3


print(
    "Vocabulary loaded."
)

print(
    "Vocabulary entries:",
    len(adjusted_word_index)
)


# ============================================================
# 4. TEXT PREPROCESSING
# ============================================================

def preprocess_review(
    review
):
    """
    Convert raw review text into the integer representation
    expected by the trained IMDB model.
    """

    # --------------------------------------------------------
    # Basic cleaning
    # --------------------------------------------------------

    review = (
        review
        .lower()
        .strip()
    )


    # Remove simple punctuation
    punctuation = (
        '.,!?;:"()[]{}'
    )


    for symbol in punctuation:

        review = review.replace(
            symbol,
            " "
        )


    # --------------------------------------------------------
    # Tokenise
    # --------------------------------------------------------

    words = review.split()


    # --------------------------------------------------------
    # Start sequence with IMDB <start> token
    # --------------------------------------------------------

    sequence = [
        1
    ]


    # --------------------------------------------------------
    # Convert words to integer IDs
    # --------------------------------------------------------

    for word in words:

        word_id = (
            adjusted_word_index.get(
                word,
                2
            )
        )


        # The model only knows IDs below VOCAB_SIZE
        if word_id >= VOCAB_SIZE:

            word_id = 2


        sequence.append(
            word_id
        )


    # --------------------------------------------------------
    # Pad to same length used during training
    # --------------------------------------------------------

    padded_sequence = pad_sequences(
        [sequence],
        maxlen=MAX_LENGTH,
        padding="pre",
        truncating="pre"
    )


    return padded_sequence


# ============================================================
# 5. PREDICTION FUNCTION
# ============================================================

def predict_sentiment(
    review
):
    """
    Predict sentiment and confidence.
    """

    processed_review = (
        preprocess_review(
            review
        )
    )


    probability = float(
        model.predict(
            processed_review,
            verbose=0
        )[0][0]
    )


    # Sigmoid output:
    #
    # close to 1 = positive
    # close to 0 = negative

    if probability >= 0.5:

        sentiment = "POSITIVE"

        confidence = probability

    else:

        sentiment = "NEGATIVE"

        confidence = (
            1 - probability
        )


    return (
        sentiment,
        confidence,
        probability
    )


# ============================================================
# 6. AUTOMATIC DEMONSTRATION
# ============================================================

print("\n" + "=" * 70)
print("AUTOMATIC DEMONSTRATION")
print("=" * 70)


demo_reviews = [

    "This movie was absolutely brilliant and I loved every minute of it.",

    "The acting was excellent and the story was fantastic.",

    "This was one of the worst movies I have ever watched.",

    "The movie was boring, disappointing and a complete waste of time."

]


for review in demo_reviews:

    sentiment, confidence, probability = (
        predict_sentiment(
            review
        )
    )


    print(
        "\nReview:"
    )

    print(
        review
    )


    print(
        "Prediction:",
        sentiment
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
# 7. INTERACTIVE SENTIMENT ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("INTERACTIVE SENTIMENT ANALYSIS")
print("=" * 70)


print(
    "Enter a movie review."
)

print(
    "Type 'quit' to exit."
)


while True:

    review = input(
        "\nReview: "
    ).strip()


    if review.lower() in [
        "quit",
        "exit",
        "q"
    ]:

        print(
            "\nSentiment analyser closed."
        )

        break


    if not review:

        print(
            "Please enter a review."
        )

        continue


    sentiment, confidence, probability = (
        predict_sentiment(
            review
        )
    )


    print(
        "\nSentiment:",
        sentiment
    )


    print(
        "Confidence:",
        f"{confidence * 100:.2f}%"
    )


    print(
        "Raw sigmoid score:",
        f"{probability:.4f}"
    )