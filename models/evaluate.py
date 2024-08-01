"""Evaluate the model on the test set."""

import torch


def evaluate(model, test_set, batch_size, device='cuda'):
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
