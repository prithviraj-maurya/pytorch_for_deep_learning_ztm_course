"""
Trains a PyTorch model using device agnostic code
"""
import os
import torch
import argparse
from torchvision import transforms
import datasetup, engine, model_builder, utils

if __name__ == "__main__":
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="Number of epochs to train the model for")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training and testing")
    parser.add_argument("--hidden_units", type=int, default=HIDDEN_UNITS, help="Number of hidden units in the model")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate for the optimizer")
    args = parser.parse_args()

    # Update hyperparameters based on command line arguments
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    HIDDEN_UNITS = args.hidden_units
    LEARNING_RATE = args.learning_rate
    # Setup directories
    train_dir = "data/pizza_steak_sushi/train"
    test_dir = "data/pizza_steak_sushi/test"
    # Setup device-agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Setup transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    # Create DataLoaders
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=transform,
        batch_size=BATCH_SIZE
    )
    # Setup model
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_shape=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)

    # Setup loss, optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Train the model
    results = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=NUM_EPOCHS,
        device=device
    )
    # Save the model
    utils.save_model(
        model=model,
        target_dir="models",
        model_name="tinyvgg_model.pth"
    )
