# ============================================================
# PYTORCH ENGLISH → GERMAN TRANSLATION DEMONSTRATION
# Seq2Seq Translation using GRU
# ============================================================

import os
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(42)
torch.manual_seed(42)

# ============================================================
# 1. CONFIGURATION
# ============================================================

print("=" * 70)
print("PYTORCH ENGLISH → GERMAN TRANSLATION DEMONSTRATION")
print("=" * 70)

print("PyTorch version:", torch.__version__)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# 2. TOKENISATION
# ============================================================

print("\n" + "=" * 70)
print("TOKENISATION")
print("=" * 70)

print(
    "Using simple whitespace tokenisation "
    "because spaCy native extensions are blocked "
    "by Windows Application Control."
)


def tokenize_en(text):
    """
    Tokenise an English sentence.

    For this small teaching demonstration,
    lowercase text and split on whitespace.
    """

    return (
        text
        .lower()
        .strip()
        .split()
    )


def tokenize_de(text):
    """
    Tokenise a German sentence.

    For this small teaching demonstration,
    lowercase text and split on whitespace.
    """

    return (
        text
        .lower()
        .strip()
        .split()
    )


# ============================================================
# 3. SPECIAL TOKENS
# ============================================================

SPECIALS = [
    "<unk>",
    "<pad>",
    "<bos>",
    "<eos>"
]


SPECIALS_DICT = {
    token: index
    for index, token
    in enumerate(SPECIALS)
}


UNK_IDX = SPECIALS_DICT["<unk>"]
PAD_IDX = SPECIALS_DICT["<pad>"]
BOS_IDX = SPECIALS_DICT["<bos>"]
EOS_IDX = SPECIALS_DICT["<eos>"]


print("\nSpecial tokens:")
print(SPECIALS_DICT)


# ============================================================
# 4. DEMONSTRATION TRAINING DATA
# ============================================================

# NOTE:
# This is deliberately a very small demonstration dataset.
# It teaches the mechanics of Seq2Seq translation.
# It is NOT sufficient for a real-world translation model.

english_sentences = [
    "hello",
    "this",
    "is",
    "a",
    "test",
    "hello this is a test"
]


german_sentences = [
    "hallo",
    "das",
    "ist",
    "ein",
    "test",
    "hallo das ist ein test"
]


print("\n" + "=" * 70)
print("DEMONSTRATION DATA")
print("=" * 70)


for english, german in zip(
    english_sentences,
    german_sentences
):

    print(
        f"{english:<25} -> {german}"
    )


# ============================================================
# 5. VOCABULARY BUILDER
# ============================================================

def build_vocab(
    sentences,
    tokenizer
):
    """
    Build a vocabulary from a list of sentences.

    Vocabulary indices 0-3 are reserved
    for the special tokens.
    """

    counter = Counter()


    for sentence in sentences:

        tokens = tokenizer(
            sentence
        )

        counter.update(
            tokens
        )


    vocabulary = {
        word: index
        for index, word
        in enumerate(
            counter.keys(),
            start=len(SPECIALS)
        )
    }


    return vocabulary


# ============================================================
# 6. BUILD ENGLISH AND GERMAN VOCABULARIES
# ============================================================

english_word_vocab = build_vocab(
    english_sentences,
    tokenize_en
)


german_word_vocab = build_vocab(
    german_sentences,
    tokenize_de
)


english_vocab = {
    **SPECIALS_DICT,
    **english_word_vocab
}


german_vocab = {
    **SPECIALS_DICT,
    **german_word_vocab
}


print("\n" + "=" * 70)
print("VOCABULARIES")
print("=" * 70)


print("\nEnglish vocabulary:")
print(english_vocab)


print("\nGerman vocabulary:")
print(german_vocab)


# ============================================================
# 7. REVERSE GERMAN VOCABULARY
# ============================================================

german_index_to_word = {
    index: word
    for word, index
    in german_vocab.items()
}


# ============================================================
# 8. TRANSLATION DATASET CLASS
# ============================================================

class TranslationDataset(Dataset):

    def __init__(
        self,
        english_sentences,
        german_sentences,
        english_vocab,
        german_vocab,
        english_tokenizer,
        german_tokenizer
    ):

        self.english_sentences = (
            english_sentences
        )

        self.german_sentences = (
            german_sentences
        )

        self.english_vocab = (
            english_vocab
        )

        self.german_vocab = (
            german_vocab
        )

        self.english_tokenizer = (
            english_tokenizer
        )

        self.german_tokenizer = (
            german_tokenizer
        )


    def __len__(self):

        return len(
            self.english_sentences
        )


    def __getitem__(
        self,
        index
    ):

        # ----------------------------------------------------
        # Tokenise source sentence
        # ----------------------------------------------------

        english_tokens = (
            self.english_tokenizer(
                self.english_sentences[
                    index
                ]
            )
        )


        # ----------------------------------------------------
        # Tokenise target sentence
        # ----------------------------------------------------

        german_tokens = (
            self.german_tokenizer(
                self.german_sentences[
                    index
                ]
            )
        )


        # ----------------------------------------------------
        # Convert English tokens to indices
        # ----------------------------------------------------

        english_indices = [
            BOS_IDX
        ]


        english_indices.extend(

            self.english_vocab.get(
                token,
                UNK_IDX
            )

            for token
            in english_tokens

        )


        english_indices.append(
            EOS_IDX
        )


        # ----------------------------------------------------
        # Convert German tokens to indices
        # ----------------------------------------------------

        german_indices = [
            BOS_IDX
        ]


        german_indices.extend(

            self.german_vocab.get(
                token,
                UNK_IDX
            )

            for token
            in german_tokens

        )


        german_indices.append(
            EOS_IDX
        )


        # ----------------------------------------------------
        # Convert lists to PyTorch tensors
        # ----------------------------------------------------

        english_tensor = torch.tensor(
            english_indices,
            dtype=torch.long
        )


        german_tensor = torch.tensor(
            german_indices,
            dtype=torch.long
        )


        return (
            english_tensor,
            german_tensor
        )


# ============================================================
# 9. PADDING FUNCTION
# ============================================================

def pad_collate(batch):
    """
    Pad variable-length sentences within a batch.

    Output shape:
    [sequence_length, batch_size]
    """

    source_sequences = [
        source
        for source, _
        in batch
    ]


    target_sequences = [
        target
        for _, target
        in batch
    ]


    source_padded = pad_sequence(
        source_sequences,
        padding_value=PAD_IDX
    )


    target_padded = pad_sequence(
        target_sequences,
        padding_value=PAD_IDX
    )


    return (
        source_padded,
        target_padded
    )


# ============================================================
# 10. ENCODER
# ============================================================

class Encoder(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size
    ):

        super().__init__()


        self.hidden_size = (
            hidden_size
        )


        # Convert word indices into dense vectors
        self.embedding = nn.Embedding(
            num_embeddings=input_size,
            embedding_dim=hidden_size,
            padding_idx=PAD_IDX
        )


        # GRU processes the source sequence
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size
        )


    def forward(
        self,
        source
    ):

        # source:
        # [source_length, batch_size]

        embedded = self.embedding(
            source
        )

        # embedded:
        # [source_length, batch_size, hidden_size]

        outputs, hidden = self.gru(
            embedded
        )


        return (
            outputs,
            hidden
        )


# ============================================================
# 11. DECODER
# ============================================================

class Decoder(nn.Module):

    def __init__(
        self,
        hidden_size,
        output_size
    ):

        super().__init__()


        self.hidden_size = (
            hidden_size
        )

        self.output_size = (
            output_size
        )


        # Embed target words
        self.embedding = nn.Embedding(
            num_embeddings=output_size,
            embedding_dim=hidden_size,
            padding_idx=PAD_IDX
        )


        # Decoder GRU
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size
        )


        # Convert GRU output into vocabulary scores
        self.output_layer = nn.Linear(
            hidden_size,
            output_size
        )


    def forward(
        self,
        decoder_input,
        hidden
    ):

        # decoder_input:
        # [1, batch_size]

        embedded = self.embedding(
            decoder_input
        )


        # Run one decoding step
        output, hidden = self.gru(
            embedded,
            hidden
        )


        # Produce raw vocabulary logits
        prediction = self.output_layer(
            output
        )


        return (
            prediction,
            hidden
        )


# ============================================================
# 12. SEQUENCE-TO-SEQUENCE MODEL
# ============================================================

class Seq2Seq(nn.Module):

    def __init__(
        self,
        encoder,
        decoder
    ):

        super().__init__()


        self.encoder = (
            encoder
        )

        self.decoder = (
            decoder
        )


    def forward(
        self,
        source,
        target,
        teacher_forcing_ratio=0.5
    ):

        # Number of target timesteps
        target_length = (
            target.shape[0]
        )


        # Number of sentences in batch
        batch_size = (
            target.shape[1]
        )


        # Number of words in German vocabulary
        target_vocab_size = (
            self.decoder.output_size
        )


        # Storage for decoder outputs
        outputs = torch.zeros(
            target_length,
            batch_size,
            target_vocab_size,
            device=source.device
        )


        # ----------------------------------------------------
        # ENCODER
        # ----------------------------------------------------

        _, hidden = self.encoder(
            source
        )


        # ----------------------------------------------------
        # Initial decoder input = <bos>
        # ----------------------------------------------------

        decoder_input = (
            target[
                0,
                :
            ]
            .unsqueeze(0)
        )


        # ----------------------------------------------------
        # Decode one word at a time
        # ----------------------------------------------------

        for time_step in range(
            1,
            target_length
        ):

            output, hidden = (
                self.decoder(
                    decoder_input,
                    hidden
                )
            )


            # output:
            # [1, batch_size, target_vocab_size]

            outputs[
                time_step
            ] = output.squeeze(0)


            # Highest-scoring predicted token
            predicted_token = (
                output
                .argmax(dim=2)
            )


            # ------------------------------------------------
            # Teacher forcing
            # ------------------------------------------------

            use_teacher_forcing = (
                random.random()
                < teacher_forcing_ratio
            )


            if use_teacher_forcing:

                decoder_input = (
                    target[
                        time_step,
                        :
                    ]
                    .unsqueeze(0)
                )

            else:

                decoder_input = (
                    predicted_token
                )


        return outputs


# ============================================================
# 13. CREATE DATASET
# ============================================================

dataset = TranslationDataset(
    english_sentences=english_sentences,
    german_sentences=german_sentences,
    english_vocab=english_vocab,
    german_vocab=german_vocab,
    english_tokenizer=tokenize_en,
    german_tokenizer=tokenize_de
)


print("\n" + "=" * 70)
print("DATASET")
print("=" * 70)


print(
    "Number of sentence pairs:",
    len(dataset)
)


# ============================================================
# 14. CREATE DATALOADER
# ============================================================

BATCH_SIZE = 1

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=pad_collate
)


# ============================================================
# 15. INSPECT ONE BATCH
# ============================================================

source_batch, target_batch = next(
    iter(dataloader)
)


print(
    "Source batch shape:",
    source_batch.shape
)


print(
    "Target batch shape:",
    target_batch.shape
)


# ============================================================
# 16. MODEL CONFIGURATION
# ============================================================

INPUT_SIZE = len(
    english_vocab
)


OUTPUT_SIZE = len(
    german_vocab
)

BATCH_SIZE = 1

HIDDEN_SIZE = 128

LEARNING_RATE = 0.001

NUM_EPOCHS = 1000

TEACHER_FORCING_RATIO = 1.0


print("\n" + "=" * 70)
print("MODEL CONFIGURATION")
print("=" * 70)


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


print(
    "Batch size:",
    BATCH_SIZE
)


print(
    "Learning rate:",
    LEARNING_RATE
)


print(
    "Teacher forcing ratio:",
    TEACHER_FORCING_RATIO
)


print(
    "Epochs:",
    NUM_EPOCHS
)


# ============================================================
# 17. CREATE ENCODER
# ============================================================

encoder = Encoder(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE
).to(
    DEVICE
)


# ============================================================
# 18. CREATE DECODER
# ============================================================

decoder = Decoder(
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_SIZE
).to(
    DEVICE
)


# ============================================================
# 19. CREATE SEQ2SEQ MODEL
# ============================================================

model = Seq2Seq(
    encoder=encoder,
    decoder=decoder
).to(
    DEVICE
)


print("\n" + "=" * 70)
print("SEQ2SEQ MODEL ARCHITECTURE")
print("=" * 70)


print(model)


# ============================================================
# 20. COUNT MODEL PARAMETERS
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


print(
    "\nTotal parameters:",
    f"{total_parameters:,}"
)


print(
    "Trainable parameters:",
    f"{trainable_parameters:,}"
)


# ============================================================
# 21. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_IDX
)


# ============================================================
# 22. OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 23. TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):

    model.train()


    total_loss = 0.0


    for source, target in dataloader:

        source = source.to(
            device
        )

        target = target.to(
            device
        )


        # ----------------------------------------------------
        # Reset gradients
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        output = model(
            source,
            target,
            teacher_forcing_ratio=(
                TEACHER_FORCING_RATIO
            )
        )


        # ----------------------------------------------------
        # Reshape for CrossEntropyLoss
        # ----------------------------------------------------

        output_dimension = (
            output.shape[-1]
        )


        # Ignore first timestep (<bos>)
        output = (
            output[1:]
            .reshape(
                -1,
                output_dimension
            )
        )


        target_for_loss = (
            target[1:]
            .reshape(-1)
        )


        # ----------------------------------------------------
        # Calculate loss
        # ----------------------------------------------------

        loss = criterion(
            output,
            target_for_loss
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        # ----------------------------------------------------
        # Update parameters
        # ----------------------------------------------------

        optimizer.step()


        total_loss += (
            loss.item()
        )


    average_loss = (
        total_loss
        / len(dataloader)
    )


    return average_loss


# ============================================================
# 24. TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)


loss_history = []


for epoch in range(
    NUM_EPOCHS
):

    loss = train_one_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE
    )


    loss_history.append(
        loss
    )


    # Print first epoch and every 100 epochs
    if (
        epoch == 0
        or (epoch + 1) % 100 == 0
        or (epoch + 1) == NUM_EPOCHS
    ):

        print(
            f"Epoch "
            f"{epoch + 1:4d}/"
            f"{NUM_EPOCHS}"
            f" | Loss: "
            f"{loss:.4f}"
        )


# ============================================================
# 25. SAVE MODEL CHECKPOINT
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "translation_model.pth"
)


checkpoint = {

    "model_state_dict":
        model.state_dict(),

    "english_vocab":
        english_vocab,

    "german_vocab":
        german_vocab,

    "input_size":
        INPUT_SIZE,

    "output_size":
        OUTPUT_SIZE,

    "hidden_size":
        HIDDEN_SIZE,

    "pad_idx":
        PAD_IDX,

    "bos_idx":
        BOS_IDX,

    "eos_idx":
        EOS_IDX,

    "unk_idx":
        UNK_IDX,

    "loss_history":
        loss_history
}


torch.save(
    checkpoint,
    MODEL_PATH
)


# ============================================================
# 26. COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)


print(
    "Model saved to:",
    MODEL_PATH
)


print(
    "Final training loss:",
    f"{loss_history[-1]:.4f}"
)


print("\nKey concepts demonstrated:")

print(
    "1. Text tokenisation"
)

print(
    "2. Vocabulary construction"
)

print(
    "3. Special tokens"
)

print(
    "4. Numericalisation"
)

print(
    "5. Padding"
)

print(
    "6. PyTorch Dataset"
)

print(
    "7. PyTorch DataLoader"
)

print(
    "8. Word embeddings"
)

print(
    "9. GRU recurrent neural networks"
)

print(
    "10. Encoder-decoder architecture"
)

print(
    "11. Sequence-to-sequence learning"
)

print(
    "12. Teacher forcing"
)

print(
    "13. Backpropagation"
)

print(
    "14. Gradient clipping"
)

print(
    "15. Model checkpoint saving"
)


print("\n" + "=" * 70)
print("PYTORCH DEMONSTRATION COMPLETE")
print("=" * 70)