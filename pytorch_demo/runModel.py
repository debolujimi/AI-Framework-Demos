# ============================================================
# PYTORCH ENGLISH → GERMAN TRANSLATION INFERENCE
# Loads the trained Seq2Seq checkpoint and translates sentences
# ============================================================

import os

import torch
import torch.nn as nn


# ============================================================
# 1. CONFIGURATION
# ============================================================

print("=" * 70)
print("PYTORCH ENGLISH → GERMAN TRANSLATOR")
print("=" * 70)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("PyTorch version:", torch.__version__)
print("Device:", DEVICE)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "translation_model.pth"
)


# ============================================================
# 2. TOKENISATION
# ============================================================

def tokenize_en(text):
    """
    Simple whitespace tokenizer.
    Must match the tokenizer used during training.
    """

    return (
        text
        .lower()
        .strip()
        .split()
    )


# ============================================================
# 3. ENCODER
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
            num_embeddings=input_size,
            embedding_dim=hidden_size,
            padding_idx=pad_idx
        )

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size
        )


    def forward(
        self,
        source
    ):

        embedded = self.embedding(
            source
        )

        outputs, hidden = self.gru(
            embedded
        )

        return outputs, hidden


# ============================================================
# 4. DECODER
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
            num_embeddings=output_size,
            embedding_dim=hidden_size,
            padding_idx=pad_idx
        )

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size
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
# 5. SEQ2SEQ MODEL
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
# 6. LOAD CHECKPOINT
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAINED MODEL")
print("=" * 70)


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


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


# Reverse German vocabulary
german_index_to_word = {
    index: word
    for word, index
    in german_vocab.items()
}


# ============================================================
# 7. REBUILD MODEL ARCHITECTURE
# ============================================================

encoder = Encoder(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    pad_idx=PAD_IDX
).to(
    DEVICE
)


decoder = Decoder(
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_SIZE,
    pad_idx=PAD_IDX
).to(
    DEVICE
)


model = Seq2Seq(
    encoder=encoder,
    decoder=decoder
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
print("Model:", MODEL_PATH)

print(
    "English vocabulary size:",
    INPUT_SIZE
)

print(
    "German vocabulary size:",
    OUTPUT_SIZE
)


# ============================================================
# 8. TRANSLATION FUNCTION
# ============================================================

def translate_sentence(
    sentence,
    model,
    english_vocab,
    german_index_to_word,
    max_length=20
):

    model.eval()


    # --------------------------------------------------------
    # Tokenise English input
    # --------------------------------------------------------

    tokens = tokenize_en(
        sentence
    )


    # --------------------------------------------------------
    # Convert English words to indices
    # --------------------------------------------------------

    source_indices = [
        BOS_IDX
    ]


    source_indices.extend(

        english_vocab.get(
            token,
            UNK_IDX
        )

        for token
        in tokens

    )


    source_indices.append(
        EOS_IDX
    )


    # --------------------------------------------------------
    # Convert to tensor
    #
    # Shape:
    # [sequence_length, batch_size]
    # --------------------------------------------------------

    source_tensor = torch.tensor(
        source_indices,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(1)


    translated_tokens = []


    with torch.no_grad():

        # ----------------------------------------------------
        # ENCODE ENGLISH SENTENCE
        # ----------------------------------------------------

        _, hidden = model.encoder(
            source_tensor
        )


        # ----------------------------------------------------
        # Start decoder with <bos>
        # ----------------------------------------------------

        decoder_input = torch.tensor(
            [[BOS_IDX]],
            dtype=torch.long,
            device=DEVICE
        )


        # ----------------------------------------------------
        # Generate German one token at a time
        # ----------------------------------------------------

        for _ in range(
            max_length
        ):

            output, hidden = (
                model.decoder(
                    decoder_input,
                    hidden
                )
            )


            predicted_index = (
                output
                .argmax(dim=2)
                .item()
            )


            # Stop when <eos> is generated
            if predicted_index == EOS_IDX:

                break


            # Ignore special tokens
            if predicted_index not in [
                PAD_IDX,
                BOS_IDX,
                UNK_IDX
            ]:

                predicted_word = (
                    german_index_to_word.get(
                        predicted_index,
                        "<unk>"
                    )
                )


                translated_tokens.append(
                    predicted_word
                )


            # Feed prediction back into decoder
            decoder_input = torch.tensor(
                [[predicted_index]],
                dtype=torch.long,
                device=DEVICE
            )


    return " ".join(
        translated_tokens
    )


# ============================================================
# 9. DEMONSTRATION TESTS
# ============================================================

print("\n" + "=" * 70)
print("AUTOMATIC DEMONSTRATION")
print("=" * 70)


test_sentences = [
    "hello",
    "this",
    "is",
    "a",
    "test",
    "hello this is a test"
]


for sentence in test_sentences:

    translation = translate_sentence(
        sentence,
        model,
        english_vocab,
        german_index_to_word
    )


    print(
        f"{sentence:<25} -> {translation}"
    )


# ============================================================
# 10. INTERACTIVE TRANSLATION
# ============================================================

print("\n" + "=" * 70)
print("INTERACTIVE TRANSLATION")
print("=" * 70)

print(
    "Enter an English sentence."
)

print(
    "Type 'quit' to exit."
)


while True:

    sentence = input(
        "\nEnglish: "
    ).strip()


    if sentence.lower() in [
        "quit",
        "exit",
        "q"
    ]:

        print(
            "\nTranslator closed."
        )

        break


    if not sentence:

        print(
            "Please enter a sentence."
        )

        continue


    translation = translate_sentence(
        sentence,
        model,
        english_vocab,
        german_index_to_word
    )


    print(
        "German:",
        translation
    )