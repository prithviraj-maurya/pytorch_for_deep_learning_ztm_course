"""
Predicts the class of an image using a pre-trained PyTorch model
"""
import torch
from torchvision import transforms
from PIL import Image
from model_builder import TinyVGG
from utils import save_model
from pathlib import Path

def predict(image_path: str, model_path: str, class_names: list):
    """
    Predicts the class of an image using a pre-trained PyTorch model.

    Args:
        image_path: Path to the image to predict.
        model_path: Path to the pre-trained model.
        class_names: List of class names.
    """
    # Load the model
    model = TinyVGG(input_shape=3, hidden_shape=10, output_shape=len(class_names))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Setup transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # Load and transform the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        y_pred = model(image)
        y_pred_class = torch.argmax(y_pred, dim=1)
        predicted_class = class_names[y_pred_class.item()]
        print(f"Predicted class: {predicted_class}")
