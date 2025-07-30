import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from models.ternary_cnn import TernaryCNN
from data.download import get_dataloaders
import random
import numpy as np


def train_model(
    log_dir: str = "./logs",
    checkpoint_dir: str = "./checkpoints",
    num_epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Train a TernaryCNN model on MNIST dataset and save logs & model.

    Args:
        log_dir (str): Directory for TensorBoard logs.
        checkpoint_dir (str): Directory to save the model checkpoint.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate.
        device (str): 'cuda' or 'cpu'
        seed (int): Random seed for reproducibility.
    """



    # Load MNIST data
    print("------------------------------Downloading the dataset..----------------------------------------")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size)

    # Initialize model, loss, optimizer
    print(f"-----------------------Using device {device}----------------------------------------")
    model = TernaryCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct / total * 100
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)

                val_loss += loss.item()
                pred = out.argmax(dim=1)
                val_correct += pred.eq(y).sum().item()
                val_total += y.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total * 100
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        # Logging
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        scheduler.step()

    # Save final model
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "ternary_cnn.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"✅ Model saved to {checkpoint_path}")

    writer.close()

    return train_losses, train_accuracies, val_losses, val_accuracies



