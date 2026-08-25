# ============================================================
# KERAS SENTIMENT ANALYSIS DEMONSTRATION
# IMDB Reviews + Pre-trained GloVe Embeddings + LSTM
# ============================================================

import os
import urllib.request
import zipfile

import numpy as np
import matplotlib.pyplot as plt

import keras

from keras import Sequential
from keras.datasets import imdb
from keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout
)
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping


# ============================================================
# 1. CONFIGURATION
# ============================================================

print("=" * 70)
print("KERAS SENTIMENT ANALYSIS DEMONSTRATION")
print("=" * 70)

print("Keras version:", keras.__version__)
print("Backend:", keras.backend.backend())


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


VOCAB_SIZE = 10000

MAX_LENGTH = 200

EMBEDDING_DIM = 100

BATCH_SIZE = 64

EPOCHS = 10


GLOVE_URL = (
    "https://nlp.stanford.edu/data/"
    "glove.6B.zip"
)


GLOVE_ZIP_PATH = os.path.join(
    BASE_DIR,
    "glove.6B.zip"
)


GLOVE_DIR = os.path.join(
    BASE_DIR,
    "glove"
)


GLOVE_FILE = os.path.join(
    GLOVE_DIR,
    "glove.6B.100d.txt"
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "keras_sentiment_model.keras"
)


# ============================================================
# 2. LOAD IMDB DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING IMDB DATASET")
print("=" * 70)


(
    x_train,
    y_train
), (
    x_test,
    y_test
) = imdb.load_data(
    num_words=VOCAB_SIZE
)


print(
    "Training reviews:",
    len(x_train)
)


print(
    "Testing reviews:",
    len(x_test)
)


# ============================================================
# 3. PAD SEQUENCES
# ============================================================

print("\n" + "=" * 70)
print("PADDING SEQUENCES")
print("=" * 70)


x_train = pad_sequences(
    x_train,
    maxlen=MAX_LENGTH,
    padding="pre",
    truncating="pre"
)


x_test = pad_sequences(
    x_test,
    maxlen=MAX_LENGTH,
    padding="pre",
    truncating="pre"
)


print(
    "Training shape:",
    x_train.shape
)


print(
    "Testing shape:",
    x_test.shape
)


# ============================================================
# 4. OBTAIN IMDB WORD INDEX
# ============================================================

print("\n" + "=" * 70)
print("BUILDING IMDB WORD INDEX")
print("=" * 70)


word_index = imdb.get_word_index()


# IMDB reserves:
# 0 = <pad>
# 1 = <start>
# 2 = <unk>
# 3 = <unused>
#
# Therefore actual word IDs must be shifted by 3.

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
    "IMDB vocabulary entries:",
    len(adjusted_word_index)
)


# ============================================================
# 5. DOWNLOAD GLOVE IF NECESSARY
# ============================================================

def download_glove():
    """
    Download and extract GloVe embeddings
    if they are not already available.
    """

    if os.path.exists(
        GLOVE_FILE
    ):

        print("\n" + "=" * 70)
        print("GLOVE EMBEDDINGS")
        print("=" * 70)

        print(
            "GloVe embeddings already available."
        )

        return


    print("\n" + "=" * 70)
    print("DOWNLOADING GLOVE EMBEDDINGS")
    print("=" * 70)


    if not os.path.exists(
        GLOVE_ZIP_PATH
    ):

        print(
            "Downloading:",
            GLOVE_URL
        )


        urllib.request.urlretrieve(
            GLOVE_URL,
            GLOVE_ZIP_PATH
        )


    else:

        print(
            "GloVe ZIP already downloaded."
        )


    os.makedirs(
        GLOVE_DIR,
        exist_ok=True
    )


    print(
        "Extracting GloVe files..."
    )


    with zipfile.ZipFile(
        GLOVE_ZIP_PATH,
        "r"
    ) as archive:

        archive.extractall(
            GLOVE_DIR
        )


    print(
        "GloVe extraction complete."
    )


# ============================================================
# 6. LOAD GLOVE EMBEDDINGS
# ============================================================

def load_glove_embeddings(
    glove_file
):

    print("\n" + "=" * 70)
    print("LOADING GLOVE EMBEDDINGS")
    print("=" * 70)


    embeddings_index = {}


    with open(
        glove_file,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            values = (
                line
                .rstrip()
                .split()
            )


            word = values[0]


            vector = np.asarray(
                values[1:],
                dtype="float32"
            )


            embeddings_index[
                word
            ] = vector


    print(
        "GloVe vectors loaded:",
        len(embeddings_index)
    )


    return embeddings_index


# ============================================================
# 7. CREATE EMBEDDING MATRIX
# ============================================================

def create_embedding_matrix(
    word_index,
    embeddings_index,
    vocab_size,
    embedding_dim
):

    print("\n" + "=" * 70)
    print("CREATING EMBEDDING MATRIX")
    print("=" * 70)


    embedding_matrix = np.zeros(
        (
            vocab_size,
            embedding_dim
        ),
        dtype="float32"
    )


    matched_words = 0


    for word, index in (
        word_index.items()
    ):

        if index >= vocab_size:

            continue


        embedding_vector = (
            embeddings_index.get(
                word
            )
        )


        if embedding_vector is not None:

            embedding_matrix[
                index
            ] = embedding_vector


            matched_words += 1


    print(
        "Vocabulary words matched to GloVe:",
        matched_words
    )


    print(
        "Embedding matrix shape:",
        embedding_matrix.shape
    )


    return embedding_matrix


# ============================================================
# 8. PREPARE GLOVE
# ============================================================

download_glove()


embeddings_index = (
    load_glove_embeddings(
        GLOVE_FILE
    )
)


embedding_matrix = (
    create_embedding_matrix(
        adjusted_word_index,
        embeddings_index,
        VOCAB_SIZE,
        EMBEDDING_DIM
    )
)


# ============================================================
# 9. BUILD KERAS MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING KERAS LSTM MODEL")
print("=" * 70)


model = Sequential([

    keras.Input(
        shape=(MAX_LENGTH,)
    ),


    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        weights=[
            embedding_matrix
        ],
        trainable=False,
        mask_zero=True
    ),


    LSTM(
        64
    ),


    Dropout(
        0.5
    ),


    Dense(
        32,
        activation="relu"
    ),


    Dropout(
        0.3
    ),


    Dense(
        1,
        activation="sigmoid"
    )

])


# ============================================================
# 10. COMPILE MODEL
# ============================================================

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]

)


model.summary()


# ============================================================
# 11. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=2,

    restore_best_weights=True

)


# ============================================================
# 12. TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING MODEL")
print("=" * 70)


history = model.fit(

    x_train,

    y_train,

    validation_split=0.2,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping
    ],

    verbose=1

)


# ============================================================
# 13. EVALUATE MODEL
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING MODEL")
print("=" * 70)


test_loss, test_accuracy = (
    model.evaluate(
        x_test,
        y_test,
        verbose=0
    )
)


print(
    "Test loss:",
    f"{test_loss:.4f}"
)


print(
    "Test accuracy:",
    f"{test_accuracy:.4f}"
)


print(
    "Test accuracy (%):",
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# 14. DISPLAY FINAL TRAINING METRICS
# ============================================================

final_train_accuracy = (
    history.history[
        "accuracy"
    ][-1]
)


final_validation_accuracy = (
    history.history[
        "val_accuracy"
    ][-1]
)


final_train_loss = (
    history.history[
        "loss"
    ][-1]
)


final_validation_loss = (
    history.history[
        "val_loss"
    ][-1]
)


print("\n" + "=" * 70)
print("FINAL TRAINING METRICS")
print("=" * 70)


print(
    "Final training accuracy:",
    f"{final_train_accuracy:.4f}"
)


print(
    "Final validation accuracy:",
    f"{final_validation_accuracy:.4f}"
)


print(
    "Final training loss:",
    f"{final_train_loss:.4f}"
)


print(
    "Final validation loss:",
    f"{final_validation_loss:.4f}"
)


print(
    "Epochs actually completed:",
    len(
        history.history[
            "loss"
        ]
    )
)


# ============================================================
# 15. PLOT ACCURACY
# ============================================================

print("\n" + "=" * 70)
print("PLOTTING TRAINING HISTORY")
print("=" * 70)


plt.figure(
    figsize=(10, 5)
)


plt.plot(
    history.history[
        "accuracy"
    ],
    label="Training Accuracy"
)


plt.plot(
    history.history[
        "val_accuracy"
    ],
    label="Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.title(
    "Keras Sentiment Analysis Accuracy"
)


plt.legend()


plt.tight_layout()


plt.show()


# ============================================================
# 16. PLOT LOSS
# ============================================================

plt.figure(
    figsize=(10, 5)
)


plt.plot(
    history.history[
        "loss"
    ],
    label="Training Loss"
)


plt.plot(
    history.history[
        "val_loss"
    ],
    label="Validation Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.title(
    "Keras Sentiment Analysis Loss"
)


plt.legend()


plt.tight_layout()


plt.show()


# ============================================================
# 17. SAVE TRAINED MODEL
# ============================================================

model.save(
    MODEL_PATH
)


print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)


print(
    "Model saved to:",
    MODEL_PATH
)


print(
    "Final test accuracy:",
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# 18. KEY CONCEPTS
# ============================================================

print("\nKey concepts demonstrated:")


print(
    "1. Text classification"
)


print(
    "2. IMDB sentiment dataset"
)


print(
    "3. Sequence padding and truncation"
)


print(
    "4. Pre-trained GloVe embeddings"
)


print(
    "5. Embedding matrix construction"
)


print(
    "6. Padding masks"
)


print(
    "7. Keras Sequential API"
)


print(
    "8. LSTM recurrent neural network"
)


print(
    "9. Dropout regularisation"
)


print(
    "10. Binary sentiment classification"
)


print(
    "11. Validation monitoring"
)


print(
    "12. Early stopping"
)


print(
    "13. Test-set evaluation"
)


print(
    "14. Model persistence"
)


print("\n" + "=" * 70)
print("KERAS DEMONSTRATION COMPLETE")
print("=" * 70)
