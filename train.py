import argparse
import os
import json
from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import ViTCropper
from dataset import get_loaders
import size_conversion


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    coord_criterion: nn.Module,
                    shape_criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> Tuple[float, float, float]:
    """
    Train the model for one epoch.

    Args:
        model: ViTCropper instance
        loader: DataLoader yielding (imgs, (coords, shape_class))
        coord_criterion: MSELoss for coordinates
        shape_criterion: CrossEntropyLoss for shape classes
        optimizer: optimizer instance
        device: torch.device

    Returns:
        Tuple of (total_loss, coord_loss, shape_loss) averages for the epoch.
    """
    model.train()
    running_loss = 0.0
    running_coord_loss = 0.0
    running_shape_loss = 0.0

    for imgs, targets in loader:
        imgs = imgs.to(device)
        coords_true = targets[0].to(device)  # [x, y] coordinates
        shape_class = targets[1].to(device)  # shape class

        optimizer.zero_grad()
        coords_pred, shape_logits = model(imgs)

        # Calculate losses
        loss_coords = coord_criterion(coords_pred, coords_true)
        loss_shape = shape_criterion(shape_logits, shape_class)
        # Use average of losses to balance their contributions
        loss = (loss_coords + loss_shape) / 2

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_coord_loss += loss_coords.item()
        running_shape_loss += loss_shape.item()

    return (running_loss / len(loader), 
            running_coord_loss / len(loader), 
            running_shape_loss / len(loader))


def eval_one_epoch(model: nn.Module,
                   loader: DataLoader,
                   coord_criterion: nn.Module,
                   shape_criterion: nn.Module,
                   device: torch.device) -> Tuple[float, float, float]:
    """
    Evaluate the model for one epoch.

    Returns:
        Tuple of (total_loss, coord_loss, shape_loss) averages for the epoch.
    """
    model.eval()
    running_loss = 0.0
    running_coord_loss = 0.0
    running_shape_loss = 0.0

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            coords_true = targets[0].to(device)  # [x, y] coordinates
            shape_class = targets[1].to(device)  # shape class

            coords_pred, shape_logits = model(imgs)

            # Calculate losses
            loss_coords = coord_criterion(coords_pred, coords_true)
            loss_shape = shape_criterion(shape_logits, shape_class)
            # Use average of losses to balance their contributions
            total_loss = (loss_coords + loss_shape) / 2
            
            running_loss += total_loss.item()
            running_coord_loss += loss_coords.item()
            running_shape_loss += loss_shape.item()

    return (running_loss / len(loader),
            running_coord_loss / len(loader),
            running_shape_loss / len(loader))


def main(project: str,
         base: str = None,
         num_epochs: int = 50,
         learning_rate: float = 1e-4,
         batch_size: int = 16,
         val_split: float = 0.2,
         patience: int = 10,
         model_name: str = 'google/vit-base-patch16-224') -> str:
    """
    Orchestrates data loading, model training, and evaluation.

    Returns:
        Path to the best saved model.
    """
    print("Starting training of ViTCropper...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
        
    labels_file = os.path.join(project, f'{base}_crop_labels.csv')
    model_path = os.path.join(project, f'{base}_crop_model.pth')
    epoch_log_file = os.path.join(project, f"{base}_crop_epoch_log.csv")
    image_dir = os.path.join(project, base)

    # Clean old artifacts
    for fp in (model_path, epoch_log_file):
        if os.path.exists(fp):
            os.remove(fp)
            print(f"Removed: {fp}")

    if not os.path.exists(epoch_log_file):
        with open(epoch_log_file, 'w') as f:
            f.write("epoch,tr_coord_loss,tr_shape_loss,val_coord_loss,val_shape_loss\n")

    # Read the CSV file to determine the unique shape classes
    try:
        # Read the labels file
        df = pd.read_csv(labels_file)
        
        # Get shape classes from the appropriate column
        if 'crop_shape' in df.columns:
            # Get unique shape classes
            if df['crop_shape'].dtype == 'object':
                # If they're strings, convert to integers using size_conversion
                shape_ints = df['crop_shape'].apply(lambda x: size_conversion.get_int_from_shape(x) if isinstance(x, str) else int(x))
            else:
                # Already integers
                shape_ints = df['crop_shape']
            
            # Find all unique class values used in the dataset
            unique_shape_classes = sorted(shape_ints.unique())
            
            # Number of output neurons equals the number of unique classes
            num_shape_classes = len(unique_shape_classes)
            
            print(f"Found {num_shape_classes} unique shape classes in dataset: {unique_shape_classes}")
            print(f"Using {num_shape_classes} output neurons for shape classification")
        else:
            print(f"Error: CSV file does not contain 'crop_shape' column")
            return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

    # Data loaders
    train_loader, val_loader = get_loaders(labels_file, image_dir,
                                           batch_size=batch_size,
                                           val_split=val_split)
    print(f"Data: {len(train_loader.dataset)} train / {len(val_loader.dataset)} val samples")

    # Initialize model and optimizer
    model = ViTCropper(pretrained_model_name=model_name,
                       num_shape_classes=num_shape_classes).to(device)
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded weights from {model_path}")

    coord_criterion = nn.MSELoss()
    shape_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=patience//2, verbose=True)

    best_coord_loss = float('inf')
    best_shape_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        
        # Train and get separate losses
        train_loss, train_coord_loss, train_shape_loss = train_one_epoch(
            model, train_loader, coord_criterion, shape_criterion, optimizer, device)
        
        # Evaluate and get separate losses
        val_loss, val_coord_loss, val_shape_loss = eval_one_epoch(
            model, val_loader, coord_criterion, shape_criterion, device)
        
        # Use the average loss from eval function for learning rate scheduler
        scheduler.step(val_loss)

        # Log all losses in a concise format
        print(f"Loss - Train: {train_loss:.4f} ({train_coord_loss:.4f}, {train_shape_loss:.4f}), Val: {val_loss:.4f} ({val_coord_loss:.4f}, {val_shape_loss:.4f})")

        
        # Write to epoch log CSV
        with open(epoch_log_file, 'a') as f:
            f.write(f"{epoch},{train_coord_loss:.6f},{train_shape_loss:.6f},{val_coord_loss:.6f},{val_shape_loss:.6f}\n")

        # Save model only if both coord and shape losses have improved independently
        coord_improved = val_coord_loss < best_coord_loss
        shape_improved = val_shape_loss < best_shape_loss
        
        if coord_improved and shape_improved:
            best_coord_loss = val_coord_loss
            best_shape_loss = val_shape_loss
            epochs_no_improve = 0
            
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model with losses - Coord: {best_coord_loss:.6f}, Shape: {best_shape_loss:.6f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

    print(f"Training complete. Best validation losses - Coord: {best_coord_loss:.6f}, Shape: {best_shape_loss:.6f}")
    return model_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ViTCropper model.")
    parser.add_argument('--project', required=True,
                        help='Root project directory containing data and labels')
    parser.add_argument('--base', type=str, default=None,
                        help='Base directory name within project containing images. If not provided, next crop_X directory will be used.')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', '-b', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--val_split', '-v', type=float, default=0.2,
                        help='Validation split fraction')
    parser.add_argument('--patience', '-p', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--model', '-m', type=str,
                        default='google/vit-base-patch16-224',
                        help='HuggingFace ViT model name')
    args = parser.parse_args()
    main(
        args.project,
        args.base,
        args.epochs,
        args.learning_rate,
        args.batch_size,
        args.val_split,
        args.patience,
        args.model
    )
