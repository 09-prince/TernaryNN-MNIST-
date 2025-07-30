import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(batch_size=128, val_split=0.1):
    """
    Prepare PyTorch DataLoaders for the MNIST dataset.

    Args:
        batch_size (int, optional): Number of samples per batch. Default is 128.
        val_split (float, optional): Fraction of training data to use for validation.
                                     Default is 0.1 (10% validation).

    Returns:
        tuple: (train_loader, val_loader, test_loader)
            - train_loader: DataLoader for the training dataset.
            - val_loader: DataLoader for the validation dataset.
            - test_loader: DataLoader for the test dataset.

    Notes:
        - The MNIST dataset will be automatically downloaded if not present locally.
        - Normalization is applied using the mean (0.1307) and std deviation (0.3081) 
          calculated from MNIST training data.
        - Training set is randomly split into training and validation subsets.
    """

    # Transform: Convert image to tensor & normalize pixel values
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL image to tensor [0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # Normalizes with MNIST mean & std
    ])

    # Download full MNIST training dataset
    full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    # Determine validation set size
    val_size = int(len(full) * val_split)

    # Randomly split into training and validation datasets
    train_ds, val_ds = random_split(full, [len(full) - val_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Load MNIST test dataset
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# Optional test run
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
