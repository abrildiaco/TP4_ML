import torch
import numpy as np
from torch.utils.data import DataLoader

def train_autoencoder(model, train_loader, eval_loader, loss_fn, optimizer, epochs, device, verbose = True):
    """
    Trains an autoencoder using reconstruction loss.
    The model receives X as input and is trained to reconstruct the same X.

    Arguments:
        model (nn.Module): autoencoder model
        train_loader (DataLoader): training DataLoader with pairs (X, X)
        eval_loader (DataLoader): evaluation DataLoader with pairs (X, X)
        loss_fn: reconstruction loss function
        optimizer: PyTorch optimizer
        epochs (int): number of training epochs
        device (torch.device): device used for computation
        verbose (bool): whether to print losses after each epoch

    Returns:
        dict: training history with train and evaluation losses
    """
    history = {
        "train_loss": [],
        "eval_loss": []
    }

    # Move the model to the selected device
    model.to(device)

    for epoch in range(epochs):
        model.train()

        total_train_loss = 0.0
        total_train_samples = 0

        for X_batch, target_batch in train_loader:
            # Move batch tensors to the selected device
            X_batch = X_batch.to(device)
            target_batch = target_batch.to(device)

            # Reconstruct the input batch
            reconstructed_batch = model(X_batch)

            # Compute reconstruction loss between reconstruction and original input
            loss = loss_fn(reconstructed_batch, target_batch)

            # Update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * X_batch.shape[0]
            total_train_samples += X_batch.shape[0]

        train_loss = total_train_loss / total_train_samples

        eval_loss = evaluate_autoencoder(model, eval_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["eval_loss"].append(eval_loss)

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} - train loss: {train_loss:.6f} - eval loss: {eval_loss:.6f}")

    return history


def evaluate_autoencoder(model, dataloader, loss_fn, device):
    """
    Evaluates an autoencoder using reconstruction loss.

    Arguments:
        model (nn.Module): autoencoder model
        dataloader (DataLoader): DataLoader with pairs (X, X)
        loss_fn: reconstruction loss function
        device (torch.device): device used for computation

    Returns:
        float: average reconstruction loss
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for X_batch, target_batch in dataloader:
            # Move batch tensors to the selected device
            X_batch = X_batch.to(device)
            target_batch = target_batch.to(device)

            # Reconstruct the input batch
            reconstructed_batch = model(X_batch)

            # Compute reconstruction loss
            loss = loss_fn(reconstructed_batch, target_batch)

            total_loss += loss.item() * X_batch.shape[0] # loss.item() is the batches mean loss
            total_samples += X_batch.shape[0]

    avg_loss = total_loss / total_samples

    return avg_loss


def reconstruct_autoencoder(model, X, device, batch_size = 256):
    """
    Reconstructs a dataset using a trained autoencoder.

    Arguments:
        model (nn.Module): trained autoencoder model
        X (pd.DataFrame | np.ndarray): input data to reconstruct
        device (torch.device): device used for computation
        batch_size (int): number of observations per batch

    Returns:
        np.ndarray: reconstructed data
    """
    # Convert X to NumPy if it comes as a pandas DataFrame
    X_values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)

    # Convert the data to a PyTorch tensor
    X_tensor = torch.tensor(X_values, dtype = torch.float32)

    # Create batches directly from the tensor
    dataloader = DataLoader(X_tensor, batch_size = batch_size, shuffle = False)

    model.eval()
    model.to(device)

    reconstructions = []

    with torch.no_grad():
        for X_batch in dataloader:
            # Move the batch to the selected device
            X_batch = X_batch.to(device)

            # Reconstruct the batch and move it back to CPU
            reconstructed_batch = model(X_batch).cpu().numpy()

            reconstructions.append(reconstructed_batch)

    # Join all reconstructed batches into one array
    X_reconstructed = np.vstack(reconstructions)

    return X_reconstructed


def encode_autoencoder(model, X, device):
    """
    Encodes a dataset using the encoder part of a trained autoencoder.
    This returns the lower-dimensional latent representation of the input data.

    Arguments:
        model (nn.Module): trained autoencoder model
        X (pd.DataFrame | np.ndarray): input data to encode
        device (torch.device): device used for computation

    Returns:
        np.ndarray: latent representation of the input data
    """
    # Convert X to NumPy if it comes as a pandas DataFrame
    X_values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)

    # Convert the data to a PyTorch tensor and moves it to the selected device
    X_tensor = torch.tensor(X_values, dtype = torch.float32).to(device)

    model.eval()
    model.to(device)

    with torch.no_grad():
        # Use only the encoder to obtain the latent representation
        X_latent = model.encode(X_tensor).cpu().numpy()

    return X_latent


def compute_reconstruction_metrics(X_original, X_reconstructed):
    """
    Computes reconstruction error metrics between original and reconstructed data.

    Arguments:
        X_original (pd.DataFrame | np.ndarray): original data
        X_reconstructed (pd.DataFrame | np.ndarray): reconstructed data

    Returns:
        dict: MSE, RMSE and MAE reconstruction metrics
    """
    # Convert inputs to NumPy arrays if needed
    X_original_values = X_original.to_numpy() if hasattr(X_original, "to_numpy") else np.asarray(X_original)
    X_reconstructed_values = X_reconstructed.to_numpy() if hasattr(X_reconstructed, "to_numpy") else np.asarray(X_reconstructed)

    mse = np.mean((X_original_values - X_reconstructed_values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(X_original_values - X_reconstructed_values))

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae
    }

    return metrics