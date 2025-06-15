import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys

from chatbot_lib.model import Transformer, save_transformer
from chatbot_lib.vocab import Vocabulary
from chatbot_lib.data import load_and_preprocess_data, vectorize_preprocessed_data, load_preprocessed_data, load_vectorized_data
from chatbot_lib.utils import get_device

from chatbot_lib.consts import (
    SRC_VOCAB_SIZE,
    TGT_VOCAB_SIZE,
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    D_FF,
    MAX_SEQ_LENGTH,
    DROPOUT,
)
from chatbot_lib.consts import BATCH_SIZE, NUM_EPOCHS, REFERENCES_PATH

# If GPU is available, set the device to GPU, otherwise use CPU
device = get_device()

def load_data(load_preprocessed: bool = False, load_vectorized: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Create a Vocabulary object
    vocab = Vocabulary()

    # Load, preprocess and vectorize the data
    if not load_vectorized:
        if load_preprocessed:
            print("Loading already preprocessed data!")
            data_df = load_preprocessed_data()
        else:
            print("Loading and preprocessing data...")
            data_df = load_and_preprocess_data()
    
    if load_vectorized:
        print("Loading already vectorized data!")
        seq_df = load_vectorized_data()
    else:
        print("Vectorizing preprocessed data...")
        seq_df = vectorize_preprocessed_data(data_df, vocab)

    # Convert the sequences to tensors
    print("Converting data to tensors...")
    seq1_tensor = torch.tensor(seq_df["seq1"].tolist(), dtype=torch.long).to(device)
    seq2_tensor = torch.tensor(seq_df["seq2"].tolist(), dtype=torch.long).to(device)

    # Split the data into training and validation sets
    train_src_data = seq1_tensor[:150000]
    train_tgt_data = seq2_tensor[:150000]
    val_src_data = seq1_tensor[150000:]
    val_tgt_data = seq2_tensor[150000:]

    # Print the shapes of the training and validation data
    print(f"Training source data shape: {train_src_data.shape}")
    print(f"Training target data shape: {train_tgt_data.shape}")
    print(f"Validation source data shape: {val_src_data.shape}")
    print(f"Validation target data shape: {val_tgt_data.shape}")

    return train_src_data, train_tgt_data, val_src_data, val_tgt_data

def train_model(transformer: Transformer, train_src_data: torch.Tensor, train_tgt_data: torch.Tensor, criterion: nn.CrossEntropyLoss, optimizer: optim.Adam, num_epochs: int, batch_size: int, vocab_size: int) -> list[float]:
    transformer.train()
    losses = []

    for epoch in range(num_epochs):
        for batch in range(batch_size):
            print(f"Epoch: {epoch+1}, batch: {batch+1}", end="\t")
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            if end_idx > len(train_src_data):
                break

            train_src_batch = train_src_data[start_idx:end_idx]
            train_tgt_batch = train_tgt_data[start_idx:end_idx]

            # Forward pass
            optimizer.zero_grad()
            output = transformer(train_src_batch, train_tgt_batch[:, :-1])
            loss = criterion(output.contiguous().view(-1, vocab_size),
                             train_tgt_batch[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            print(f"Loss: {loss.item()}")

    return losses

def evaluate_model(transformer: Transformer, val_src_data: torch.Tensor, val_tgt_data: torch.Tensor, criterion: nn.CrossEntropyLoss, vocab_size) -> list[float]:
    transformer.eval()
    losses = []

    with torch.no_grad():
        for batch in range(BATCH_SIZE):
            print(f"Batch: {batch+1}", end="\t")
            start_idx = batch * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            if end_idx > len(val_src_data):
                break

            val_src_batch = val_src_data[start_idx:end_idx]
            val_tgt_batch = val_tgt_data[start_idx:end_idx]

            val_output = transformer(val_src_batch, val_tgt_batch[:, :-1])
            val_loss = criterion(val_output.contiguous().view(-1, vocab_size), val_tgt_batch[:, 1:].contiguous().view(-1))

            losses.append(val_loss.item())
            print(f"Validation Loss: {val_loss.item()}")

    return losses

def save_plot(losses: list[float], val_losses: list[float], filepath):
    plt.plot(losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'{REFERENCES_PATH}/loss_plot.png')
    plt.close()

def main():
    load_preprocessed = ("-p" in sys.argv)
    load_vectorized = ("-v" in sys.argv)

    # Load and preprocess the data
    train_src_data, train_tgt_data, val_src_data, val_tgt_data = load_data(load_preprocessed, load_vectorized)
    print("Data loaded successfully.")

    # Initialize the Transformer model, optimizer, and loss function
    print("Initializing the Transformer model, optimizer, and loss function...")
    transformer = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, MAX_SEQ_LENGTH, DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    print("Starting training...")
    losses = train_model(transformer, train_src_data, train_tgt_data, criterion, optimizer, NUM_EPOCHS, BATCH_SIZE, TGT_VOCAB_SIZE)
    
    print("Starting evaluation...")
    val_losses = evaluate_model(transformer, val_src_data, val_tgt_data, criterion, TGT_VOCAB_SIZE)
    
    filepath = f"{REFERENCES_PATH}/loss_plot.png"
    save_plot(losses, val_losses, filepath)
    print(f"Training complete. Losses saved to {filepath}")

    save_transformer(transformer)
    print(f"Model saved!")

if __name__ == "__main__":
    main()
