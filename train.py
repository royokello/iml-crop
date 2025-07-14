import argparse
import os
import json
from pathlib import Path
import torch
import torch.nn as nn
from typing import Tuple
from torch.utils.data import DataLoader

from model import IMLCropModel
from dataset import get_loaders
from utils import find_latest_stage
from torch.optim.lr_scheduler import OneCycleLR

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    coord_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device
) -> float:
    
    model.train()
    running_coord_loss = 0.0

    for imgs, targets in loader:
        imgs = imgs.to(device)
        coords_true, _ = targets
        coords_true = coords_true.to(device)

        optimizer.zero_grad()
        coords_pred, _ = model(imgs)

        loss_coords = coord_criterion(coords_pred, coords_true)
        loss_coords.backward()
        optimizer.step()
        scheduler.step()

        running_coord_loss += loss_coords.item()

    return running_coord_loss / len(loader)


def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    coord_criterion: nn.Module,
    device: torch.device
) -> float:
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


def main(
    project: str,
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    val_split: float,
    stage: int = None
):
    print("training started ...")

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
    models_dir = os.path.join(project, f"{stage_dir}_crop_models")
    os.makedirs(models_dir, exist_ok=True)
    epoch_log_file = os.path.join(models_dir, f"{stage_dir}_crop_epoch_log.csv")
    image_dir = os.path.join(project, stage_dir)

    with open(epoch_log_file, 'w') as f:
        f.write("epoch,train_coord_loss,val_coord_loss,loss_gap\n")

    if not os.path.exists(ratios_file):
        raise FileNotFoundError(f"Ratio definitions not found: {ratios_file}")
    with open(ratios_file, 'r') as rf:
        ratios = json.load(rf)
    num_ratio_classes = len(ratios)
    print(f"Model will use {num_ratio_classes} ratio classes")

    train_loader, val_loader = get_loaders(
        samples_file,
        image_dir,
        batch_size=batch_size,
        val_split=val_split
    )

    model = IMLCropModel(
        pretrained_model_name="google/vit-base-patch16-384",
        num_ratio_classes=num_ratio_classes
    ).to(device)

    epoch_files = sorted(
        Path(models_dir).glob("epoch_*.pth"),
        key=lambda p: int(p.stem.split("_")[1])
    )
    resume_epoch = 0
    if epoch_files:
        latest = epoch_files[-1]
        resume_epoch = int(latest.stem.split("_")[1])
        model.load_state_dict(torch.load(latest, map_location=device))
        print(f"Resuming from epoch {resume_epoch} (loaded {latest.name})")

    coord_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=100.0
    )

    best_val_loss = float('inf')

    print("training loop started ...")
    for epoch in range(resume_epoch + 1, num_epochs + 1):
        
        train_loss = train_one_epoch(model, train_loader, coord_criterion, optimizer, scheduler, device)
        val_loss = eval_one_epoch(model, val_loader, coord_criterion, device)

        loss_gap = abs(val_loss - train_loss)

        print(f"epoch {epoch}/{num_epochs}, tr. coord loss: {train_loss:.8f}, val. coord Loss: {val_loss:.8f}, gap: {loss_gap:.8f}")

        with open(epoch_log_file, 'a') as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.8f},{loss_gap:.8f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(models_dir, f"epoch_{epoch}.pth")
            torch.save(model.state_dict(), model_path)

    print(f"training complete!")
    return model_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--stage', type=int, default=None)
    parser.add_argument('--epochs', '-e', type=int, default=64)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', '-b', type=int, default=10)
    parser.add_argument('--val_split', '-v', type=float, default=0.2)
    args = parser.parse_args()
    main(
        project=args.project,
        stage=args.stage,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        val_split=args.val_split,
    )
