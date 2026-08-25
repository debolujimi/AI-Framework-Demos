# ============================================================
# PYTORCH SEQ2SEQ MODEL EVALUATION
# English → German Translation
# ============================================================

import os
import time

import torch
import torch.nn as nn


# ============================================================
# 1. CONFIGURATION
# ============================================================

print("=" * 75)
print("PYTORCH ENGLISH → GERMAN MODEL EVALUATION")
print("=" * 75)

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
    Tokenizer must match the tokenizer used during training.
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

print("\n" + "=" * 75)
print("LOADING CHECKPOINT")
print("=" * 75)

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
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


german_index_to_word = {
    index: word
    for word, index
    in german_vocab.items()
}


print("Checkpoint loaded successfully.")

print(
    "English vocabulary size:",
    INPUT_SIZE
)

print(
    "German vocabulary size:",
    OUTPUT_SIZE
)

print(
    "Hidden size:",
    HIDDEN_SIZE
)


# ============================================================
# 7. REBUILD MODEL
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


# ============================================================
# 8. PARAMETER COUNT
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter
    in model.parameters()
)


trainable_parameters = sum(
    parameter.numel()
    for parameter
    in model.parameters()
    if parameter.requires_grad
)


# ============================================================
# 9. MODEL FILE SIZE
# ============================================================

model_size_mb = (
    os.path.getsize(
        MODEL_PATH
    )
    / (1024 * 1024)
)


# ============================================================
# 10. TRANSLATION FUNCTION
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

        for token
        in tokens

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

                predicted_word = (
                    german_index_to_word.get(
                        predicted_index,
                        "<unk>"
                    )
                )

                translated_tokens.append(
                    predicted_word
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
# 11. KNOWN TRAINING EXAMPLES
# ============================================================

known_examples = [

    (
        "hello",
        "hallo"
    ),

    (
        "this",
        "das"
    ),

    (
        "is",
        "ist"
    ),

    (
        "a",
        "ein"
    ),

    (
        "test",
        "test"
    ),

    (
        "hello this is a test",
        "hallo das ist ein test"
    )

]


# ============================================================
# 12. EXACT-MATCH EVALUATION
# ============================================================

print("\n" + "=" * 75)
print("KNOWN-SENTENCE EVALUATION")
print("=" * 75)


exact_matches = 0

total_examples = len(
    known_examples
)


predicted_sentences = []


for english_sentence, expected_german in known_examples:

    predicted_german = translate_sentence(
        english_sentence
    )


    predicted_sentences.append(
        predicted_german
    )


    is_correct = (
        predicted_german.strip()
        == expected_german.strip()
    )


    if is_correct:

        exact_matches += 1


    status = (
        "CORRECT"
        if is_correct
        else "INCORRECT"
    )


    print(
        f"\nEnglish:   "
        f"{english_sentence}"
    )

    print(
        f"Expected:  "
        f"{expected_german}"
    )

    print(
        f"Predicted: "
        f"{predicted_german}"
    )

    print(
        f"Result:    "
        f"{status}"
    )


sentence_accuracy = (
    exact_matches
    / total_examples
)


# ============================================================
# 13. TOKEN-LEVEL ACCURACY
# ============================================================

correct_tokens = 0
total_reference_tokens = 0


for (
    (_, expected_german),
    predicted_german
) in zip(
    known_examples,
    predicted_sentences
):

    expected_tokens = (
        expected_german.split()
    )

    predicted_tokens = (
        predicted_german.split()
    )


    total_reference_tokens += len(
        expected_tokens
    )


    for index, expected_token in enumerate(
        expected_tokens
    ):

        if (
            index < len(predicted_tokens)
            and predicted_tokens[index]
            == expected_token
        ):

            correct_tokens += 1


token_accuracy = (
    correct_tokens
    / total_reference_tokens
    if total_reference_tokens > 0
    else 0.0
)


# ============================================================
# 14. INFERENCE LATENCY
# ============================================================

print("\n" + "=" * 75)
print("INFERENCE LATENCY")
print("=" * 75)


benchmark_sentence = (
    "hello this is a test"
)


# Warm-up runs
for _ in range(10):

    translate_sentence(
        benchmark_sentence
    )


NUMBER_OF_RUNS = 100


start_time = time.perf_counter()


for _ in range(
    NUMBER_OF_RUNS
):

    translate_sentence(
        benchmark_sentence
    )


end_time = time.perf_counter()


total_inference_time = (
    end_time
    - start_time
)


average_latency_seconds = (
    total_inference_time
    / NUMBER_OF_RUNS
)


average_latency_ms = (
    average_latency_seconds
    * 1000
)


print(
    "Benchmark sentence:",
    benchmark_sentence
)

print(
    "Number of measured runs:",
    NUMBER_OF_RUNS
)

print(
    "Average inference latency:",
    f"{average_latency_ms:.3f} ms"
)


# ============================================================
# 15. UNSEEN SENTENCE DEMONSTRATION
# ============================================================

print("\n" + "=" * 75)
print("UNSEEN-SENTENCE DEMONSTRATION")
print("=" * 75)


unseen_sentences = [

    "this is a test",

    "hello test",

    "a test",

    "hello this",

    "good morning"

]


for sentence in unseen_sentences:

    translation = translate_sentence(
        sentence
    )


    print(
        f"{sentence:<25}"
        f" -> "
        f"{translation}"
    )


# ============================================================
# 16. TRAINING LOSS
# ============================================================

loss_history = checkpoint.get(
    "loss_history",
    []
)


if loss_history:

    initial_loss = (
        loss_history[0]
    )

    final_loss = (
        loss_history[-1]
    )

else:

    initial_loss = None
    final_loss = None


# ============================================================
# 17. FINAL EVALUATION SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL MODEL EVALUATION")
print("=" * 75)


print(
    f"{'Metric':<35}"
    f"{'Result':>20}"
)

print("-" * 55)


print(
    f"{'Known sentence pairs':<35}"
    f"{total_examples:>20}"
)


print(
    f"{'Exact matches':<35}"
    f"{exact_matches:>20}"
)


print(
    f"{'Sentence exact-match accuracy':<35}"
    f"{sentence_accuracy * 100:>19.2f}%"
)


print(
    f"{'Token-level accuracy':<35}"
    f"{token_accuracy * 100:>19.2f}%"
)


print(
    f"{'Total parameters':<35}"
    f"{total_parameters:>20,}"
)


print(
    f"{'Trainable parameters':<35}"
    f"{trainable_parameters:>20,}"
)


print(
    f"{'Checkpoint size':<35}"
    f"{model_size_mb:>17.3f} MB"
)


print(
    f"{'Average inference latency':<35}"
    f"{average_latency_ms:>17.3f} ms"
)


if initial_loss is not None:

    print(
        f"{'Initial training loss':<35}"
        f"{initial_loss:>20.6f}"
    )


if final_loss is not None:

    print(
        f"{'Final training loss':<35}"
        f"{final_loss:>20.6f}"
    )


print("\n" + "=" * 75)
print("IMPORTANT INTERPRETATION")
print("=" * 75)

print(
    "The exact-match and token accuracy above are measured "
    "on the six sentences used for training."
)

print(
    "They therefore measure memorisation of the demonstration "
    "dataset, not generalisation to unseen language."
)

print(
    "The unseen-sentence section is qualitative and should "
    "not be treated as a formal test-set accuracy."
)


print("\nEvaluation complete.")