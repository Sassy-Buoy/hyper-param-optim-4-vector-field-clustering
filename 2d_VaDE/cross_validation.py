"""train_model, cross_val, and evaluate_model functions for training and evaluating autoencoders."""

import torch
from sklearn.model_selection import KFold


import torch
import torch.nn.functional as F


def train_model(model, train_set, val_set, lr, batch_size, epochs, patience=5, device='cuda'):
    """Train the autoencoder with early stopping based on validation loss stability."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

    model.to(device)
    model.train()

    best_val_loss = float('inf')
    epochs_since_improvement = 0

    for epoch in range(epochs):
        running_loss = 0.0

        # Training phase
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.get_loss(batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                loss = model.get_loss(batch)
                val_loss += loss.item()

        epoch_val_loss = val_loss / len(val_loader)

        if torch.isnan(torch.tensor(epoch_val_loss)):
            print("Validation loss is nan. Stopping training.")
            return None

        print(
            f"Epoch {epoch+1}/{epochs} Train Loss: {epoch_train_loss:.4f} Val Loss: {epoch_val_loss:.4f}")

        # Check for early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_since_improvement = 0
            # Optionally, you might want to save the best model here
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(
                    f"Early stopping triggered. No improvement in validation loss for {patience} epochs.")
                break

        model.train()  # Switch back to training mode after validation


def cross_val(model, train_set, lr, batch_size, epochs, n_splits=5, device='cuda'):
    """Perform cross-validation on the autoencoder."""
    torch.backends.cudnn.benchmark = True
    kf = KFold(n_splits=n_splits, shuffle=True)
    val_losses = []

    for fold, (train_index, val_index) in enumerate(kf.split(train_set)):
        print(f"Fold {fold+1}/{n_splits}")

        train_sampler = torch.utils.data.SubsetRandomSampler(train_index)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_index)

        train_loader = torch.utils.data.DataLoader(
            train_set, sampler=train_sampler, batch_size=batch_size, num_workers=12, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            train_set, sampler=val_sampler, batch_size=batch_size, num_workers=12, pin_memory=True)

        model.to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = model.get_loss(batch)
                # stop training if loss is nan
                if torch.isnan(loss):
                    print("Loss is nan. Stopping training.")
                    return None
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_sampler)
            print(f"Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.4f}")

        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                loss = model.get_loss(batch)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_sampler)
        print(f"Validation Loss: {avg_loss:.4f}")
        val_losses.append(avg_loss)

    print("fin.")
    return val_losses


def evaluate_model(model, test_set, batch_size, device='cuda'):
    """Evaluate the autoencoder."""
    dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=12)
    model.to(device)
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            loss = model.get_loss(batch)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_set)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss
