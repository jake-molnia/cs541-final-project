import os
from re import S
import numpy as np
import pickle
from typing import Tuple, List, Dict, Any, Optional
import matplotlib.pyplot as plt
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear, Embedding, Conv1d, Conv2d
from tinygrad.nn.optim import AdamW
from tinygrad import Device
import wandb
import time

# Hyperparameters
BATCH_SIZE: int = 64
EPOCHS: int = 10
LEARNING_RATE: float = 0.001
EMBEDDING_DIM: int = 100
WEIGHT_DECAY: float = 1e-5
DROPOUT_RATE: float = 0.2
CNN_FILTERS: List[int] = [128, 256, 512]
CNN_KERNEL_SIZES: List[int] = [3, 5, 7]
LATENT_DIMS: List[int] = [256, 128, 64]

def load_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Load preprocessed data.

    Args:
        data_dir: Directory containing preprocessed data files

    Returns:
        X_train, X_test, y_train, y_test, vocab_size, max_sequence_length
    """
    X_train: np.ndarray = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test: np.ndarray = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train: np.ndarray = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test: np.ndarray = np.load(os.path.join(data_dir, 'y_test.npy'))

    with open(os.path.join(data_dir, 'tokenizer.pickle'), 'rb') as handle:
        tokenizer: Any = pickle.load(handle)

    vocab_size: int = len(tokenizer.word_index) + 1  # +1 for padding token
    max_sequence_length: int = X_train.shape[1]

    print(f"Loaded data - X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Vocabulary size: {vocab_size}, Max sequence length: {max_sequence_length}")

    return X_train, X_test, y_train, y_test, vocab_size, max_sequence_length

class CNNAutoencoder:
    def __init__(self, vocab_size: int, embedding_dim: int,
                 cnn_filters: List[int], cnn_kernel_sizes: List[int],
                 latent_dims: List[int]) -> None:
        # Embedding layer
        self.embedding = Embedding(vocab_size, embedding_dim)

        # Multiple CNN layers
        self.convs = []
        in_channels = embedding_dim
        for i, (filters, kernel_size) in enumerate(zip(cnn_filters, cnn_kernel_sizes)):
            self.convs.append(Conv1d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2))
            in_channels = filters

        self.encoders = []
        self.decoders = []
        encoder_input_dim = cnn_filters[-1]
        # Add encoder layers
        for latent_dim in latent_dims:
            self.encoders.append(Linear(encoder_input_dim, latent_dim))
            encoder_input_dim = latent_dim
        # Add decoder layers
        decoder_input_dim = latent_dims[-1]
        for i in range(len(latent_dims)-1, 0, -1):
            self.decoders.append(Linear(decoder_input_dim, latent_dims[i-1]))
            decoder_input_dim = latent_dims[i-1]
        self.decoders.append(Linear(decoder_input_dim, cnn_filters[-1]))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        x = x.permute(0, 2, 1)
        for conv in self.convs:
            x = conv(x).relu()
        features = x.max(axis=2)  # (batch_size, cnn_filters[-1])
        encoded = features
        for encoder in self.encoders:
            encoded = encoder(encoded).relu()
        decoded = encoded
        for decoder in self.decoders:
            decoded = decoder(decoded).relu()

        return encoded, decoded, features

    def parameters(self) -> List[Tensor]:
        params = [self.embedding.weight]
        for conv in self.convs:
            params.extend([conv.weight, conv.bias])
        for encoder in self.encoders:
            params.extend([encoder.weight, encoder.bias])
        for decoder in self.decoders:
            params.extend([decoder.weight, decoder.bias])

        return params


def train(model: CNNAutoencoder, X_train: np.ndarray, X_test: np.ndarray,
         batch_size: int, epochs: int, learning_rate: float, weight_decay: float) -> CNNAutoencoder:
    """Train the model and track with Weights & Biases with live pyplot visualization."""
    # Setup for live plotting
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    train_losses, val_losses = [], []
    epochs_plot = []

    wandb.init(project="tinygrad-email-autoencoder",
        config={
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "embedding_dim": EMBEDDING_DIM,
            "cnn_filters": CNN_FILTERS,
            "cnn_kernel_sizes": CNN_KERNEL_SIZES,
            "latent_dims": LATENT_DIMS,
            "dropout_rate": DROPOUT_RATE,
            "device": Device.DEFAULT,
            "dataset_size": len(X_train),
            "test_size": len(X_test),
            "vocab_size": model.embedding.weight.shape[0],
            "max_sequence_length": X_train.shape[1]
        },
        name=f"run-bs{batch_size}-lr{learning_rate}-wd{weight_decay}",
        tags=["autoencoder", "cnn", "tinygrad"])
    batch_table = wandb.Table(columns=["epoch", "batch", "loss", "time"])
    start_time = time.time()
    epoch_start_time = start_time
    batch_start_time = start_time

    Tensor.training = True
    optimizer: AdamW = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    n_batches: int = int(np.floor(0.1 * len(X_train)) // batch_size)
    total_batches = epochs * n_batches

    global_step = 0
    batch_losses = []
    running_loss = 0.0
    custom_x = []

    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss: float = 0.0
        batch_losses = []  # Reset for new epoch

        indices: np.ndarray = np.random.permutation(len(X_train))
        X_train_shuffled: np.ndarray = X_train[indices]

        for i in range(n_batches):
            batch_start_time = time.time()
            global_step += 1
            start_idx: int = i * batch_size
            end_idx: int = start_idx + batch_size
            x_batch: np.ndarray = X_train_shuffled[start_idx:end_idx]
            x_tensor: Tensor = Tensor(x_batch)

            encoded, decoded, features = model.forward(x_tensor)
            loss: Tensor = ((features - decoded) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.detach().numpy()
            batch_time = time.time() - batch_start_time
            total_loss += batch_loss
            batch_losses.append(batch_loss)
            running_loss += batch_loss
            custom_x.append(global_step / total_batches)  # Normalized position
            wandb.log({
                "batch": global_step,
                "batch_loss": batch_loss,
                "batch_time": batch_time,
                "epoch": epoch,
                "progress": global_step / total_batches,
                "learning_rate": learning_rate,
                "estimated_time_remaining": (time.time() - start_time) / global_step * (total_batches - global_step)
            }, step=global_step)
            batch_table.add_data(epoch, i, batch_loss, batch_time)

            if i % 5 == 0 or i == n_batches - 1:
                avg_running_loss = running_loss / min(5, i+1)
                running_loss = 0.0

                ax2.clear()
                ax2.plot(range(len(batch_losses)), batch_losses, 'b-')
                ax2.set_title(f'Epoch {epoch+1}/{epochs}, Batch Losses')
                ax2.set_xlabel('Batch')
                ax2.set_ylabel('Loss')
                plt.tight_layout()
                plt.pause(0.01)  # Brief pause to update the plot

                wandb.log({
                    "smoothed_batch_loss": avg_running_loss,
                    "batch_time_moving_avg": batch_time  # Can be improved with actual moving avg
                }, step=global_step)

                if i % 10 == 0:  # Reduce frequency of image logging to avoid WandB overhead
                    plt.figure(figsize=(10, 4))
                    plt.plot(range(len(batch_losses)), batch_losses, 'b-', label='Batch Loss')
                    plt.title(f'Epoch {epoch+1}, Batch Losses')
                    plt.xlabel('Batch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.tight_layout()
                    wandb.log({"batch_loss_plot": wandb.Image(plt)}, step=global_step)
                    plt.close()

        avg_loss: float = total_loss / n_batches
        epoch_time = time.time() - epoch_start_time
        train_losses.append(avg_loss)
        epochs_plot.append(epoch)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "epoch_time": epoch_time,
            "time_elapsed": time.time() - start_time
        }, step=global_step)

        val_indices: np.ndarray = np.random.choice(len(X_test), min(1000, len(X_test)), replace=False)
        X_val: np.ndarray = X_test[val_indices]
        val_batches: int = max(1, len(X_val) // batch_size)
        val_loss: float = 0.0
        val_batch_losses = []

        val_start_time = time.time()
        for i in range(val_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(X_val))
            x_batch = X_val[start_idx:end_idx]
            x_tensor = Tensor(x_batch)

            with Tensor.no_grad():
                encoded, decoded, features = model.forward(x_tensor)
                batch_loss: float = ((features - decoded) ** 2).mean().numpy()
            val_loss += batch_loss
            val_batch_losses.append(batch_loss)

        avg_val_loss: float = val_loss / val_batches
        val_time = time.time() - val_start_time
        val_losses.append(avg_val_loss)

        print(f"Validation Loss: {avg_val_loss:.4f}, Time: {val_time:.2f}s")
        wandb.log({
            "val_loss": avg_val_loss,
            "val_time": val_time,
            "val_batch_losses": wandb.Histogram(val_batch_losses)
        }, step=global_step)

        if epoch % 2 == 0 or epoch == epochs - 1:
            grad_norms = []
            weight_norms = []
            for j, param in enumerate(model.parameters()):
                if param.grad is not None:
                    grad_norm = np.linalg.norm(param.grad.numpy())
                    grad_norms.append(grad_norm)
                    wandb.log({f"grad_norm/layer_{j}": grad_norm}, step=global_step)

                weight_norm = np.linalg.norm(param.numpy())
                weight_norms.append(weight_norm)
                wandb.log({f"weight_norm/layer_{j}": weight_norm}, step=global_step)

            wandb.log({
                "grad_norms": wandb.Histogram(grad_norms),
                "weight_norms": wandb.Histogram(weight_norms),
                "max_grad": max(grad_norms) if grad_norms else 0,
                "min_grad": min(grad_norms) if grad_norms else 0,
                "avg_grad": sum(grad_norms) / len(grad_norms) if grad_norms else 0
            }, step=global_step)

        # Update the training/validation loss plot
        ax1.clear()
        ax1.plot(epochs_plot, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs_plot, val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training Progress')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        plt.tight_layout()
        plt.pause(0.1)  # Pause to update the plot

        if epoch % 2 == 0 or epoch == epochs - 1:
            checkpoint_path = f"checkpoints/epoch_{epoch}.npz"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            save_model(model, checkpoint_path)
            checkpoint_artifact = wandb.Artifact(f"model-checkpoint-epoch-{epoch}", type="model")
            checkpoint_artifact.add_file(checkpoint_path)
            wandb.log_artifact(checkpoint_artifact)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs_plot, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs_plot, val_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        wandb.log({"loss_plot": wandb.Image(plt)}, step=global_step)
        plt.close()

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")

    wandb.log({"batch_metrics": batch_table})

    # Save final plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_plot, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs_plot, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(len(batch_losses)), batch_losses, 'g-')
    plt.title('Final Epoch Batch Losses')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_progress.png')
    wandb.log({"final_plot": wandb.Image('training_progress.png')})

    plt.ioff()  # Turn off interactive mode
    plt.close()

    Tensor.training = False
    wandb.finish()
    return model

def save_model(model: CNNAutoencoder, save_path: str) -> None:
    """Save model parameters."""
    params: Dict[str, np.ndarray] = {f"param_{i}": p.numpy() for i, p in enumerate(model.parameters())}
    np.savez(save_path, **params)
    print(f"Model saved to {save_path}")

def main() -> None:
    print(f"Using device: {Device.DEFAULT}")
    os.makedirs("data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    data_dir: str = "data"
    X_train, X_test, y_train, y_test, vocab_size, max_sequence_length = load_data(data_dir)
    model: CNNAutoencoder = CNNAutoencoder(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        cnn_filters=CNN_FILTERS,
        cnn_kernel_sizes=CNN_KERNEL_SIZES,
        latent_dims=LATENT_DIMS
    )
    train_indices = np.random.choice(len(X_train), int(0.1 * len(X_train)), replace=False)
    X_train_subset = X_train[train_indices]
    model = train(
        model=model,
        X_train=X_train_subset,
        X_test=X_test,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    save_model(model, os.path.join(data_dir, "email_autoencoder.npz"))
    print("Training complete!")

if __name__ == "__main__":
    main()
