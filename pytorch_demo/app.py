# ============================================================
# PYTORCH ENGLISH → GERMAN TRANSLATION WEB APPLICATION
# ============================================================

import os

import torch
import torch.nn as nn

from flask import Flask, render_template, request, jsonify


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
    "translation_model.pth"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# TOKENISATION
# ============================================================

def tokenize_en(text):

    return (
        text
        .lower()
        .strip()
        .split()
    )


# ============================================================
# ENCODER
# ============================================================

class Encoder(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        pad_idx
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            input_size,
            hidden_size,
            padding_idx=pad_idx
        )

        self.gru = nn.GRU(
            hidden_size,
            hidden_size
        )


    def forward(self, source):

        embedded = self.embedding(
            source
        )

        outputs, hidden = self.gru(
            embedded
        )

        return outputs, hidden


# ============================================================
# DECODER
# ============================================================

class Decoder(nn.Module):

    def __init__(
        self,
        hidden_size,
        output_size,
        pad_idx
    ):

        super().__init__()

        self.output_size = output_size

        self.embedding = nn.Embedding(
            output_size,
            hidden_size,
            padding_idx=pad_idx
        )

        self.gru = nn.GRU(
            hidden_size,
            hidden_size
        )

        self.output_layer = nn.Linear(
            hidden_size,
            output_size
        )


    def forward(
        self,
        decoder_input,
        hidden
    ):

        embedded = self.embedding(
            decoder_input
        )

        output, hidden = self.gru(
            embedded,
            hidden
        )

        prediction = self.output_layer(
            output
        )

        return prediction, hidden


# ============================================================
# SEQ2SEQ
# ============================================================

class Seq2Seq(nn.Module):

    def __init__(
        self,
        encoder,
        decoder
    ):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder


# ============================================================
# LOAD MODEL CHECKPOINT
# ============================================================

print("=" * 70)
print("LOADING PYTORCH TRANSLATION MODEL")
print("=" * 70)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

english_vocab = checkpoint[
    "english_vocab"
]

german_vocab = checkpoint[
    "german_vocab"
]

INPUT_SIZE = checkpoint[
    "input_size"
]

OUTPUT_SIZE = checkpoint[
    "output_size"
]

HIDDEN_SIZE = checkpoint[
    "hidden_size"
]

PAD_IDX = checkpoint[
    "pad_idx"
]

BOS_IDX = checkpoint[
    "bos_idx"
]

EOS_IDX = checkpoint[
    "eos_idx"
]

UNK_IDX = checkpoint[
    "unk_idx"
]


german_index_to_word = {
    index: word
    for word, index
    in german_vocab.items()
}


encoder = Encoder(
    INPUT_SIZE,
    HIDDEN_SIZE,
    PAD_IDX
).to(
    DEVICE
)

decoder = Decoder(
    HIDDEN_SIZE,
    OUTPUT_SIZE,
    PAD_IDX
).to(
    DEVICE
)

model = Seq2Seq(
    encoder,
    decoder
).to(
    DEVICE
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()

print("Model loaded successfully.")
print("Device:", DEVICE)


# ============================================================
# TRANSLATION FUNCTION
# ============================================================

def translate_sentence(
    sentence,
    max_length=20
):

    tokens = tokenize_en(
        sentence
    )

    source_indices = [
        BOS_IDX
    ]

    source_indices.extend(

        english_vocab.get(
            token,
            UNK_IDX
        )

        for token in tokens
    )

    source_indices.append(
        EOS_IDX
    )


    source_tensor = torch.tensor(
        source_indices,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(1)


    translated_tokens = []


    with torch.no_grad():

        _, hidden = model.encoder(
            source_tensor
        )

        decoder_input = torch.tensor(
            [[BOS_IDX]],
            dtype=torch.long,
            device=DEVICE
        )


        for _ in range(
            max_length
        ):

            output, hidden = model.decoder(
                decoder_input,
                hidden
            )

            predicted_index = (
                output
                .argmax(dim=2)
                .item()
            )


            if predicted_index == EOS_IDX:

                break


            if predicted_index not in [
                PAD_IDX,
                BOS_IDX,
                UNK_IDX
            ]:

                word = (
                    german_index_to_word.get(
                        predicted_index,
                        "<unk>"
                    )
                )

                translated_tokens.append(
                    word
                )


            decoder_input = torch.tensor(
                [[predicted_index]],
                dtype=torch.long,
                device=DEVICE
            )


    return " ".join(
        translated_tokens
    )


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH CHECK
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
            "pytorch-translation-ai",

        "framework":
            "PyTorch",

        "model":
            "GRU Seq2Seq"

    }), 200


# ============================================================
# TRANSLATION API
# ============================================================

@app.route(
    "/translate",
    methods=["POST"]
)
def translate():

    data = request.get_json(
        silent=True
    )


    if not data:

        return jsonify({
            "error":
                "No JSON request provided"
        }), 400


    sentence = (
        data
        .get(
            "sentence",
            ""
        )
        .strip()
    )


    if not sentence:

        return jsonify({
            "error":
                "No English sentence provided"
        }), 400


    translation = translate_sentence(
        sentence
    )


    return jsonify({

        "english":
            sentence,

        "german":
            translation

    }), 200


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("STARTING PYTORCH TRANSLATION API")
    print("=" * 70)

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False
    )