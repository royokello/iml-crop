import argparse
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
import os
from dataset import ImageDataset
from utils import log_print, setup_logging
from PIL import Image
from model import ViTCropper
import time


def find_album_directories(root_dir: str) -> list:
    albums = []
    for item in os.listdir(root_dir):
        album_dir = os.path.join(root_dir, item)
        if os.path.isdir(album_dir) and os.path.exists(os.path.join(album_dir, 'crop_labels.csv')):
            albums.append(album_dir)
    
    return albums


# Custom collate function for ViT model to handle PIL images
def vit_collate_fn(batch):
    """
    Custom collate function for ViT model that handles PIL images.
    
    Args:
        batch: List of tuples (image, target)
    
    Returns:
        tuple (images, targets) where images is a list of PIL Images and 
        targets is a tensor of bounding box coordinates in the format [x1, y1, height, ratio]
        where ratio is encoded as 0 (1:1), 1 (2:3), or 2 (3:2)
    """
    images = [item[0] for item in batch]  # Keep as PIL images
    targets = torch.stack([item[1] for item in batch])
    return images, targets


def main(root_dir: str, num_epochs: int, learning_rate: float = 0.0001, batch_size: int = 16, 
         val_split: float = 0.2, patience: int = 8, model_name: str = 'google/vit-base-patch16-224'):
    """
    Train a ViT model for image cropping on labeled image datasets.
    
    Args:
        root_dir: Root directory containing album directories
        num_epochs: Maximum number of epochs to train
        learning_rate: Learning rate for the optimizer
        batch_size: Batch size for training
        val_split: Fraction of data to use for validation
        patience: Number of epochs with no improvement before early stopping
        model_name: Pretrained ViT model name from Hugging Face
    """
    setup_logging(root_dir)
    log_print("Training ViT Cropper model started...")
    
    # Check for cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_print(f"Using device: {device}")
    
    # Find all album directories with label data
    album_dirs = find_album_directories(root_dir)
    log_print(f"Found {len(album_dirs)} album directories with label data")
    
    if len(album_dirs) == 0:
        log_print("No valid album directories found. Exiting.")
        return
    
    # Create individual datasets for each album directory
    datasets = []
    for album_dir in album_dirs:
        src_culled_dir = os.path.join(album_dir, 'src_culled')
        labels_file = os.path.join(album_dir, 'crop_labels.csv')
        
        # For ViT we don't need transformations as they're handled internally
        dataset = ImageDataset(src_culled_dir, labels_file, transform=None)
        
        # Add only if dataset contains images
        if len(dataset) > 0:
            datasets.append(dataset)
            log_print(f"Added {len(dataset)} images from {os.path.basename(album_dir)}")
    
    # Combine all datasets
    if not datasets:
        log_print("No valid datasets found. Exiting.")
        return
    
    combined_dataset = ConcatDataset(datasets)
    log_print(f"Combined dataset contains {len(combined_dataset)} images")
    
    # Split into training and validation sets
    train_size = int(len(combined_dataset) * (1 - val_split))
    val_size = len(combined_dataset) - train_size
    
    # Generate random indices for splitting
    indices = torch.randperm(len(combined_dataset)).tolist()
    train_dataset = torch.utils.data.Subset(combined_dataset, indices[:train_size])
    val_dataset = torch.utils.data.Subset(combined_dataset, indices[train_size:])
    
    log_print(f"Split dataset into {train_size} training and {val_size} validation images")
    
    # Create dataloaders with appropriate collate_fn
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=vit_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=vit_collate_fn
    )
    
    # Define model path and tracking files
    model_path = os.path.join(root_dir, 'crop_model.pth')
    epoch_file = os.path.join(root_dir, 'crop_epoch.txt')
    val_loss_file = os.path.join(root_dir, 'crop_val_loss.txt')
    
    # Load previous epoch and val_loss if available
    start_epoch = 0
    best_val_loss = float('inf')
    
    if os.path.exists(epoch_file):
        with open(epoch_file, 'r') as f:
            try:
                start_epoch = int(f.read().strip())
                log_print(f"Resuming from epoch {start_epoch}")
            except ValueError:
                log_print("Invalid epoch file content, starting from epoch 0")
    
    if os.path.exists(val_loss_file):
        with open(val_loss_file, 'r') as f:
            try:
                best_val_loss = float(f.read().strip())
                log_print(f"Previous best validation loss: {best_val_loss:.4f}")
            except ValueError:
                log_print("Invalid val_loss file content, using infinity")
    
    try:
        # Load pre-trained ViT cropper model
        model = ViTCropper(pretrained_model_name=model_name)
        model.to(device)
        
        # Load previous model weights if available
        if os.path.exists(model_path):
            log_print(f"Loading weights from existing model: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Explicitly set model to training mode
        model.train()
        
        # Define loss function and optimizer
        # For the new format, we need two loss components:
        # 1. MSE loss for coordinate regression (x1, y1, height)
        # 2. Cross-entropy loss for aspect ratio classification (0, 1, 2)
        coord_criterion = torch.nn.MSELoss()
        ratio_criterion = torch.nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=patience//2, verbose=True
        )
        
        # Training loop
        epochs_no_improve = 0
        
        log_print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            # Training phase
            model.train()  # Make sure model is in training mode
            train_loss = 0.0
            train_coord_loss = 0.0
            train_ratio_loss = 0.0
            
            for images, targets in train_loader:
                # Move targets to device
                targets = targets.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(images)
                
                # When in training mode, the model returns a tuple (coords, ratio_logits)
                if isinstance(outputs, tuple):
                    coord_preds, ratio_logits = outputs
                else:
                    # Handle unexpected case (should not happen with the fixed model)
                    if hasattr(outputs, 'shape'):
                        if outputs.shape[-1] == 3:  # Only coordinate predictions
                            coord_preds = outputs
                            batch_size = coord_preds.shape[0]
                            ratio_logits = torch.zeros((batch_size, 3), device=device)
                        else:
                            # Try to split the tensor
                            coord_preds = outputs[:, :3]
                            ratio_logits = outputs[:, 3:]
                    else:
                        raise ValueError("Unexpected output type from model")
                
                # Extract regression targets (x1, y1, height) and classification targets (ratio code)
                coord_targets = targets[:, :3].float()
                ratio_targets = targets[:, 3].long()
                
                # Calculate losses
                coord_loss = coord_criterion(coord_preds, coord_targets)
                ratio_loss = ratio_criterion(ratio_logits, ratio_targets)
                
                # Combined loss (you can adjust weights if needed)
                loss = coord_loss + ratio_loss
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update running losses
                train_loss += loss.item()
                train_coord_loss += coord_loss.item()
                train_ratio_loss += ratio_loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_train_coord_loss = train_coord_loss / len(train_loader)
            avg_train_ratio_loss = train_ratio_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_coord_loss = 0.0
            val_ratio_loss = 0.0
            
            with torch.no_grad():
                for images, targets in val_loader:
                    # Move targets to device
                    targets = targets.to(device)
                    
                    # Extract regression targets (x1, y1, height) and classification targets (ratio code)
                    coord_targets = targets[:, :3].float()
                    ratio_targets = targets[:, 3].long()
                    
                    # Forward pass
                    outputs = model(images)
                    
                    # Handle model outputs
                    if isinstance(outputs, tuple):
                        coord_preds, ratio_logits = outputs
                    else:
                        # Handle unexpected case (should not happen with the fixed model)
                        if hasattr(outputs, 'shape'):
                            if outputs.shape[-1] == 3:
                                coord_preds = outputs
                                batch_size = coord_preds.shape[0]
                                ratio_logits = torch.zeros((batch_size, 3), device=device)
                            else:
                                coord_preds = outputs[:, :3]
                                ratio_logits = outputs[:, 3:]
                        else:
                            raise ValueError("Unexpected output type from model")
                    
                    # Calculate losses
                    coord_loss = coord_criterion(coord_preds, coord_targets)
                    ratio_loss = ratio_criterion(ratio_logits, ratio_targets)
                    
                    # Combined loss
                    loss = coord_loss + ratio_loss
                    
                    # Update running losses
                    val_loss += loss.item()
                    val_coord_loss += coord_loss.item()
                    val_ratio_loss += ratio_loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            avg_val_coord_loss = val_coord_loss / len(val_loader)
            avg_val_ratio_loss = val_ratio_loss / len(val_loader)
            
            # Update learning rate scheduler
            scheduler.step(avg_val_loss)
            
            # Log progress
            log_print(f"Epoch {epoch+1}/{start_epoch + num_epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} (Coord: {avg_train_coord_loss:.4f}, Ratio: {avg_train_ratio_loss:.4f}) | "
                      f"Val Loss: {avg_val_loss:.4f} (Coord: {avg_val_coord_loss:.4f}, Ratio: {avg_val_ratio_loss:.4f})")
            
            # Save best model and check early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                
                # Save the best model
                torch.save(model.state_dict(), model_path)
                log_print(f"Saved best model with validation loss: {best_val_loss:.4f}")
                
                # Update tracking files
                with open(epoch_file, 'w') as f:
                    f.write(str(epoch + 1))
                with open(val_loss_file, 'w') as f:
                    f.write(str(best_val_loss))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    log_print(f"Early stopping after {epoch+1} epochs with no improvement")
                    break
        
        log_print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
        log_print(f"Best model saved to: {model_path}")
    
    except Exception as e:
        log_print(f"Error training model: {e}")
        log_print("Exiting.")
        return
    
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ViT cropper model on multiple album directories.")
    parser.add_argument("root_directory", type=str, help="Root directory containing album directories.")
    parser.add_argument("--epochs", "-e", type=int, default=256, help="Number of epochs to train.")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--batch_size", "-b", type=int, default=16, help="Batch size.")
    parser.add_argument("--val_split", "-v", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--patience", "-p", type=int, default=8, help="Early stopping patience.")
    parser.add_argument("--model", "-m", type=str, default='google/vit-base-patch16-224', 
                        help="Pretrained model name from Hugging Face.")
    
    args = parser.parse_args()
    
    main(
        args.root_directory, 
        args.epochs, 
        args.learning_rate, 
        args.batch_size, 
        args.val_split, 
        args.patience,
        args.model
    )