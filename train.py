import argparse
import os
import json
import torch
import torch.nn as nn
from typing import Tuple
from torch.utils.data import DataLoader

from model import IMLCropModel
from dataset import get_loaders
from utils import find_latest_stage


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    coord_criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> float:
    """
    Train the model for one epoch, tracking only coordinate loss.

    Returns:
        avg_coord_loss
    """
    model.train()
    running_coord_loss = 0.0

    for imgs, targets in loader:
        imgs = imgs.to(device)
        coords_true, _ = targets  # Ignore ratio targets
        coords_true = coords_true.to(device)

        optimizer.zero_grad()
        coords_pred, _ = model(imgs)  # Discard ratio logits

        loss_coords = coord_criterion(coords_pred, coords_true)
        loss_coords.backward()
        optimizer.step()

        running_coord_loss += loss_coords.item()

    return running_coord_loss / len(loader)


def eval_one_epoch(model: nn.Module,
                   loader: DataLoader,
                   coord_criterion: nn.Module,
                   device: torch.device) -> float:
    """
    Evaluate the model for one epoch, tracking only coordinate loss.

    Returns:
        avg_coord_loss
    """
    model.eval()
    running_coord_loss = 0.0

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            coords_true, _ = targets
            coords_true = coords_true.to(device)

            coords_pred, _ = model(imgs)
            loss_coords = coord_criterion(coords_pred, coords_true)
            running_coord_loss += loss_coords.item()

    return running_coord_loss / len(loader)


def main(project: str,
         stage: int = None,
         num_epochs: int = 50,
         learning_rate: float = 1e-4,
         batch_size: int = 16,
         val_split: float = 0.2,
         patience: int = 10,
         model_name: str = 'google/vit-base-patch16-224') -> str:
    """
    Orchestrates data loading, model training, and evaluation, focusing on coordinate loss only.
    Labels CSV must have normalized x, y, width, and integer 'ratio' columns.
    Requires a JSON file listing ratio classes for model initialization.
    """
    print("Starting training of IMLCropModel (coord only)...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if stage is None:
        try:
            stage = find_latest_stage(project)
            print(f"Using latest stage: {stage}")
        except ValueError as e:
            print(f"Error: {e}")
            return

    stage_dir = f"stage_{stage}"
    samples_file = os.path.join(project, f"{stage_dir}_crop_labels.csv")
    ratios_file = os.path.join(project, f"{stage_dir}_crop_ratios.json")
    model_path = os.path.join(project, f"{stage_dir}_crop_model.pth")
    epoch_log_file = os.path.join(project, f"{stage_dir}_crop_epoch_log.csv")
    image_dir = os.path.join(project, stage_dir)

    # Clean old artifacts
    for fp in (model_path, epoch_log_file):
        if os.path.exists(fp):
            os.remove(fp)
            print(f"Removed: {fp}")

    # Initialize log
    with open(epoch_log_file, 'w') as f:
        f.write("epoch,train_coord_loss,val_coord_loss\n")

    # Load ratio classes for model head (unused in loss)
    if not os.path.exists(ratios_file):
        raise FileNotFoundError(f"Ratio definitions not found: {ratios_file}")
    with open(ratios_file, 'r') as rf:
        ratios = json.load(rf)
    num_ratio_classes = len(ratios)
    print(f"Model will use {num_ratio_classes} ratio classes (not tracked in loss)")

    # Data loaders
    train_loader, val_loader = get_loaders(
        samples_file,
        image_dir,
        batch_size=batch_size,
        val_split=val_split
    )

    # Model setup
    model = IMLCropModel(pretrained_model_name=model_name,
                         num_ratio_classes=num_ratio_classes).to(device)
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded weights from {model_path}")

    coord_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=patience//2, verbose=True)

    best_coord = float('inf')
    no_improve = 0

    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        tr_coord = train_one_epoch(
            model, train_loader, coord_criterion, optimizer, device)
        val_coord = eval_one_epoch(
            model, val_loader, coord_criterion, device)

        scheduler.step(val_coord)

        print(f"Train Coord Loss: {tr_coord:.4f}")
        print(f"Val   Coord Loss: {val_coord:.4f}")

        # Log metrics
        with open(epoch_log_file, 'a') as f:
            f.write(f"{epoch},{tr_coord:.6f},{val_coord:.6f}\n")

        gap_abs  = abs(tr_coord - val_coord)
        gap_rel  = gap_abs / (val_coord + 1e-8)

        improved_enough = val_coord < best_coord * 0.9
        gap_ok = gap_rel <= 0.2

        if (val_coord < best_coord) and (gap_ok or improved_enough):
            best_coord = val_coord
            no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model (coord: {best_coord:.6f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered")
                break

    print(f"Training complete. Best coord: {best_coord:.6f}")
    return model_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train IMLCropModel model (coord only).")
    parser.add_argument('--project', required=True,
                        help='Root project directory containing data and labels')
    parser.add_argument('--stage', type=int, default=None,
                        help='Stage number to train (e.g., 1, 2, etc.). If not provided, uses the latest stage.')
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
                        default='google/vit-base-patch16-384',
                        help='HuggingFace ViT model name')
    args = parser.parse_args()
    main(
        args.project,
        args.stage,
        args.epochs,
        args.learning_rate,
        args.batch_size,
        args.val_split,
        args.patience,
        args.model
    )
