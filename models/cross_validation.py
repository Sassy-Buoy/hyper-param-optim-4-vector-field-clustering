"""train_model, cross_val, and evaluate_model functions for training and evaluating autoencoders."""

import torch
from sklearn.model_selection import KFold


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
