"""
Contains functionality for creating PyTorch DataLoaders for
image classification data
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_dir: str, 
                       test_dir: str, 
                       transform: transforms.Compose, 
                       batch_size: int, 
                       num_workers: int = NUM_WORKERS):
    """
    Creates training and testing DataLoaders.

    Takes in training and testing directory paths and turns them into PyTorch
    Datasets and then into DataLoaders.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision.transforms.Compose containing transformations to perform on data.
        batch_size: Number of samples per batch of data.
        num_workers: Number of workers for DataLoader to use.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
    """
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    class_names = train_data.classes
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataloader, test_dataloader, class_names
