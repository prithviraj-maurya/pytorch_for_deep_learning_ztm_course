"""
Contains functions for training and testing a PyTorch model
"""
import torch
from tqdm import tqdm
from typing import Dict, Tuple, List

def train_step(model: torch.nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                loss_fn: torch.nn.Module, 
                optimizer: torch.optim.Optimizer, 
                device: str) -> Tuple[float, float]:
    """
    Trains a PyTorch model for a single epoch.

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader containing the training data.
        loss_fn: A loss function to compute the loss.
        optimizer: An optimizer to update the model's parameters.
        device: The device to use for training (e.g., "cuda" or "cpu").

    Returns:
        A tuple of (train_loss, train_accuracy).
    """
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(y_pred, dim=1)
        train_acc += (y_pred_class == y).sum().item()
    train_loss /= len(dataloader)
    train_acc /= len(dataloader.dataset)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module, 
              device: str) -> Tuple[float, float]:
    """ Tests a PyTorch model for a single epoch.
    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader containing the testing data.
        loss_fn: A loss function to compute the loss.
        device: The device to use for testing (e.g., "cuda" or "cpu").
    Returns:
        A tuple of (test_loss, test_accuracy).
    """
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            y_pred_class = torch.argmax(y_pred, dim=1)
            test_acc += (y_pred_class == y).sum().item()
    test_loss /= len(dataloader)
    test_acc /= len(dataloader.dataset)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
        train_dataloader: torch.utils.data.DataLoader, 
        test_dataloader: torch.utils.data.DataLoader, 
        optimizer_fn: torch.nn.Module, 
        loss_fn: torch.optim.Optimizer, 
        epochs: int, 
        device: str) -> Dict[str, List[float]]:
    """ Trains and tests a PyTorch model for a specified number of epochs.
    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader containing the training data.
        test_dataloader: A DataLoader containing the testing data.
        loss_fn: A loss function to compute the loss.
        optimizer: An optimizer to update the model's parameters.
        epochs: The number of epochs to train the model for.
        device: The device to use for training and testing (e.g., "cuda" or "cpu").
    Returns:
        A dictionary of training and testing losses and accuracies.
    """
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)
        print(f"Epoch: {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    
        print(
            f"Epoch: {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f}"
            
        )
    return results
