import sys
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.ternary_cnn import TernaryCNN
from data.download import get_dataloaders
import random
from matplotlib import pyplot as plt

device: str = "cuda" if torch.cuda.is_available() else "cpu"
model_path: str = "./checkpoints/ternary_cnn.pth"
model = TernaryCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

def visualize_predictions(model, device, num_samples=20):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    indices = random.sample(range(len(test_dataset)), num_samples)
    fig, axs = plt.subplots(2, 10, figsize=(18, 5))  # 2 rows x 10 columns
    axs = axs.flatten()

    model.eval()
    for i, idx in enumerate(indices):
        img, label = test_dataset[idx]
        img_input = img.unsqueeze(0).to(device)
        pred = model(img_input).argmax(1).item()

        axs[i].imshow(img.squeeze(), cmap='gray')
        axs[i].set_title(f"GT: {label}\nPred: {pred}", fontsize=10)
        axs[i].axis("off")

    # Hide any unused axes
    for j in range(num_samples, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_predictions(model, device, num_samples=20)
