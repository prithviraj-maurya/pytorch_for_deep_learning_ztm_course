"""
Contains utility functions for saving and loading models
"""
import torch
from pathlib import Path

def save_model(model: torch.nn.Module, 
              target_dir: str, 
              model_name: str):
    """
    Saves a PyTorch model to a target directory.

    Args:
        model: A PyTorch model to be saved.
        target_dir: The directory to save the model to.
        model_name: The name of the model file.
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    model_path = target_dir_path / model_name
    print(f"Saving model to {model_path}")
    torch.save(obj=model.state_dict(), f=model_path)
