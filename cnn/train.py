import argparse
import os
import json
import csv
import torch
import torch.nn as nn
from torch.optim import Adam
from dataset import get_loaders
from model import create_model


class WeightedMSELoss(nn.Module):
    def __init__(self, coord_weight=1.0, size_weight=0.5, ratio_weight=0.5):
        super().__init__()
        self.coord_weight = coord_weight
        self.size_weight = size_weight
        self.ratio_weight = ratio_weight

    def forward(self, outputs, targets):
        pred_coords = outputs[:, :2]
        pred_size   = outputs[:, 2:3]
        pred_ratio  = outputs[:, 3:4]

        target_coords = targets[:, :2]
        target_size   = targets[:, 2:3]
        target_ratio  = targets[:, 3:4]

        coord_loss = torch.mean((pred_coords - target_coords) ** 2)
        size_loss  = torch.mean((pred_size   - target_size)   ** 2)
        ratio_loss = torch.mean((pred_ratio  - target_ratio)  ** 2)

        return (
            self.coord_weight * coord_loss +
            self.size_weight  * size_loss  +
            self.ratio_weight * ratio_loss
        )


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total += loss.item() * imgs.size(0)
    return total / len(loader.dataset)


def val_epoch(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            total += criterion(outputs, targets).item() * imgs.size(0)
    return total / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description='Train crop regression model')
    parser.add_argument('--input', required=True, help='Root dir containing labels and images')
    parser.add_argument('--source', default='src', help='Image subfolder (default: src)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--coord_weight', type=float, default=1.0, help='Weight for coordinate loss')
    parser.add_argument('--size_weight', type=float, default=0.5, help='Weight for size loss')
    parser.add_argument('--ratio_weight', type=float, default=0.5, help='Weight for ratio loss')
    parser.add_argument('--loss_window', type=int, default=5, help='Window size for plateau detection')
    parser.add_argument('--delta', type=float, default=1e-3, help='Min relative change to reset patience')
    args = parser.parse_args()

    model_path = os.path.join(args.input, 'crop_model.pth')
    info_path  = os.path.join(args.input, 'crop_model.json')
    log_path   = os.path.join(args.input, 'crop_epoch_log.csv')

    # Remove old outputs
    for fp in (model_path, info_path, log_path):
        if os.path.exists(fp):
            os.remove(fp); print(f"Removed: {fp}")

    csv_path = os.path.join(args.input, 'crop_labels.csv')
    if not os.path.exists(csv_path):
        print(f"Error: crop_labels.csv missing in {args.input}"); return
    img_dir = os.path.join(args.input, args.source)
    train_loader, val_loader = get_loaders(csv_path, img_dir, args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model().to(device)
    criterion = WeightedMSELoss(
        coord_weight=args.coord_weight,
        size_weight=args.size_weight,
        ratio_weight=args.ratio_weight
    )
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    loss_history = []

    # CSV header
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss'])

    for epoch in range(1, args.epochs+1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = val_epoch(model, val_loader, criterion, device)
        loss_history.append(train_loss)
        print(f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}"])

        # Save best by val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            info = {
                'base_model': type(model[0]).__name__,
                'epochs': epoch,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'lr': args.lr,
                'batch_size': args.batch_size,
                'coord_weight': args.coord_weight,
                'size_weight': args.size_weight,
                'ratio_weight': args.ratio_weight
            }
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=4)
            print(f"Saved model (val_loss={val_loss:.6f})")

        # Early stop on plateau of train loss
        if len(loss_history) >= args.loss_window:
            recent = loss_history[-args.loss_window:]
            diffs = [recent[i]-recent[i-1] for i in range(1,len(recent))]
            avg_rel_change = sum(diffs)/len(diffs) / (sum(recent)/len(recent))
            if avg_rel_change > -args.delta:
                print(f"Stopping early at epoch {epoch}: train loss plateau")
                break

if __name__ == '__main__':
    main()
