# ============================================================
# KERAS SENTIMENT ANALYSIS WEB APPLICATION
# ============================================================

import os

import keras

from flask import (
    Flask,
    render_template,
    request,
    jsonify
)

from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

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
# LOAD TRAINED MODEL
# ============================================================

print("=" * 70)
print("LOADING KERAS SENTIMENT MODEL")
print("=" * 70)


model = keras.models.load_model(
    MODEL_PATH
)


print("Model loaded successfully.")

print(
    "Model:",
    MODEL_PATH
)


# ============================================================
# LOAD IMDB WORD INDEX
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


print(
    "Vocabulary loaded:",
    len(adjusted_word_index)
)


# ============================================================
# PREPROCESS REVIEW
# ============================================================

def preprocess_review(review):
    """
    Convert raw review text into the integer sequence
    expected by the trained Keras model.
    """

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


    # Begin with the IMDB <start> token
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


    padded_sequence = pad_sequences(

        [sequence],

        maxlen=MAX_LENGTH,

        padding="pre",

        truncating="pre"

    )


    return padded_sequence


# ============================================================
# SENTIMENT PREDICTION
# ============================================================

def predict_sentiment(review):

    processed_review = preprocess_review(
        review
    )


    probability = float(

        model.predict(
            processed_review,
            verbose=0
        )[0][0]

    )


    if probability >= 0.5:

        sentiment = "POSITIVE"

        confidence = probability

    else:

        sentiment = "NEGATIVE"

        confidence = (
            1 - probability
        )


    return {

        "sentiment":
            sentiment,

        "confidence":
            confidence,

        "score":
            probability

    }


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "healthy",

        "service":
            "keras-sentiment-ai",

        "framework":
            "Keras",

        "backend":
            keras.backend.backend(),

        "model":
            "GloVe LSTM Sentiment Classifier"

    }), 200


# ============================================================
# PREDICT ENDPOINT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    data = request.get_json(
        silent=True
    )


    if not data:

        return jsonify({

            "error":
                "No JSON request provided"

        }), 400


    review = (
        data
        .get(
            "review",
            ""
        )
        .strip()
    )


    if not review:

        return jsonify({

            "error":
                "No review provided"

        }), 400


    result = predict_sentiment(
        review
    )


    return jsonify({

        "review":
            review,

        "sentiment":
            result["sentiment"],

        "confidence":
            result["confidence"],

        "score":
            result["score"]

    }), 200


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("STARTING KERAS SENTIMENT API")
    print("=" * 70)


    app.run(

        host="127.0.0.1",

        port=5002,

        debug=False

    )