"""train_model, cross_val, and evaluate_model functions for training and evaluating autoencoders."""

import torch
from sklearn.model_selection import KFold


def train_model(model, train_set, batch_size=64, epochs=100, lr=1e-3):
    """Train the autoencoder."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=12)
    model.to('cuda')
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for batch in dataloader:
            batch = batch.to('cuda')
            optimizer.zero_grad()
            loss = model.get_loss(batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.size(0)

        epoch_loss = running_loss / len(train_set)
        print(f"Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.4f}")


def cross_val(model, train_set, n_splits=5, batch_size=64, epochs=100, lr=1e-3):
    """Perform cross-validation on the autoencoder."""
    torch.backends.cudnn.benchmark = True
    kf = KFold(n_splits=n_splits, shuffle=True)
    val_losses = []

    for fold, (train_index, val_index) in enumerate(kf.split(train_set)):
        print(f"Fold {fold+1}/{n_splits}")

        train_sampler = torch.utils.data.SubsetRandomSampler(train_index)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_index)

        train_loader = torch.utils.data.DataLoader(
            train_set, sampler=train_sampler, batch_size=batch_size, num_workers=12)
        val_loader = torch.utils.data.DataLoader(
            train_set, sampler=val_sampler, batch_size=batch_size, num_workers=12)

        model.to('cuda')
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch in train_loader:
                batch = batch.to('cuda')
                optimizer.zero_grad()
                loss = model.get_loss(batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch.size(0)

            epoch_loss = running_loss / len(train_sampler)
            print(f"Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.4f}")

        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to('cuda')
                loss = model.get_loss(batch)
                total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(val_sampler)
        print(f"Validation Loss: {avg_loss:.4f}")
        val_losses.append(avg_loss)

    return val_losses


def evaluate_model(model, test_set, batch_size=64):
    """Evaluate the autoencoder."""
    dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=12)
    model.to('cuda')
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to('cuda')
            loss = model.get_loss(batch)
            total_loss += loss.item() * batch.size(0)

    avg_loss = total_loss / len(test_set)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss
