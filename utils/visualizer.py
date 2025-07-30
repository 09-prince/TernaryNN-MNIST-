"""
visualizer.py

This module provides visualization utilities for training and evaluating
a TernaryCNN model on the MNIST dataset.

Functions:
- plot_training_curve: Plots and saves the training/validation loss and accuracy curves.
- plot_weight_distribution: Plots and saves the distribution of ternary weights from the trained model.

All plots are saved in the specified `logs/` directory (default: logs/).
"""

import os
import torch
import matplotlib.pyplot as plt
from models.ternary_cnn import TernaryCNN


def plot_training_curve(train_losses, val_losses, train_accuracies, val_accuracies, save_dir="logs"):
    """
    Plot and save training and validation loss and accuracy curves.

    Args:
        train_losses (list of float): Training loss values per epoch.
        val_losses (list of float): Validation loss values per epoch.
        train_accuracies (list of float): Training accuracy values per epoch.
        val_accuracies (list of float): Validation accuracy values per epoch.
        save_dir (str): Directory to save the plots (default: "logs").

    Saves:
        - training_curves.png: Side-by-side plots for loss and accuracy.
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 4))

    # Plot loss curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Val Loss", color="orange")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy", color="blue", linestyle="--")
    plt.plot(val_accuracies, label="Val Accuracy", color="green")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Saved training curves to: {plot_path}")


def plot_weight_distribution(model_path="checkpoints/ternary_cnn.pth", save_dir="logs"):
    """
    Plot and save the histogram of ternarized weights from the trained TernaryCNN model.

    Args:
        model_path (str): Path to the saved model checkpoint (.pth file).
        save_dir (str): Directory to save the histogram plot (default: "logs").

    Saves:
        - ternary_weight_distribution.png: Bar plot showing count of -1, 0, and +1 weights.
    """
    os.makedirs(save_dir, exist_ok=True)

    model = TernaryCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    ternary_counts = {-1: 0, 0: 0, 1: 0}

    def ternarize_tensor(tensor, ratio=0.3):
        """
        Ternarize a given weight tensor using a threshold based on mean absolute weight.

        Args:
            tensor (torch.Tensor): The weight tensor to ternarize.
            ratio (float): The ratio used to calculate the threshold.

        Returns:
            torch.Tensor: A ternary tensor containing only -1, 0, and 1 values.
        """
        scale = tensor.abs().mean()
        thr = scale * ratio
        return torch.where(tensor > thr, 1.0,
               torch.where(tensor < -thr, -1.0, 0.0))

    # Collect ternary weights
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            with torch.no_grad():
                ternary_w = ternarize_tensor(param)
                values, counts = torch.unique(ternary_w, return_counts=True)
                for v, c in zip(values.cpu().numpy(), counts.cpu().numpy()):
                    ternary_counts[int(v)] += int(c)

    # Plot the ternary weight distribution
    labels = ['-1', '0', '+1']
    counts = [ternary_counts[-1], ternary_counts[0], ternary_counts[1]]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, counts, color=['red', 'gray', 'green'])
    plt.title("Ternary Weight Distribution")
    plt.ylabel("Number of Weights")

    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + max(counts) * 0.02,
                 f"{counts[i]}", ha='center', fontsize=10)

    out_path = os.path.join(save_dir, "ternary_weight_distribution.png")
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Saved ternary distribution plot to: {out_path}")


if __name__ == "__main__":
    # Example usage with dummy data
    train_losses = [0.8, 0.6, 0.4, 0.3]
    val_losses = [0.9, 0.65, 0.5, 0.35]
    train_accuracies = [70, 80, 90, 95]
    val_accuracies = [68, 78, 88, 92]

    plot_training_curve(train_losses, val_losses, train_accuracies, val_accuracies)
    plot_weight_distribution()
