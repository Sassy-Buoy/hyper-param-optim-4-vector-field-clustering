"""Training loop for the autoencoder model."""

import torch
import matplotlib.pyplot as plt


def train(model, train_set, val_set, lr, batch_size, epochs,
          patience=5, device='cuda'):
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
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
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

        print(f"""Epoch {epoch+1}/{epochs}
              Train Loss: {epoch_train_loss:.4f}
              Val Loss: {epoch_val_loss:.4f}""")
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model = model.state_dict()
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"""Early stopping triggered.
                      No improvement in validation loss for {patience} epochs.""")
                break

    model.load_state_dict(best_model)

    # plot the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
