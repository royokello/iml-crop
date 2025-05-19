import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import json

# Import our size conversion module
import size_conversion

def resize_and_pad_square(image, target_size=224):
    """
    Resize an image to a square while maintaining aspect ratio and padding.
    
    This function handles both portrait and landscape images:
      - For landscape images, the width is resized to target_size and the image is centered vertically with padding on top and bottom.
      - For portrait images (or square images), the height is resized to target_size and the image is centered horizontally with padding on the sides.
    
    Args:
        image (PIL.Image): Input image.
        target_size (int): Desired dimension (width and height) for the square output.
    
    Returns:
        tuple: A tuple containing:
            - square_image (PIL.Image): The resized and padded square image.
            - padding (tuple): The applied padding in the format (padding_x, padding_y)
    """
    width, height = image.size
    
    if width >= height:
        # Landscape image (or square) - scale width to target_size
        scale = target_size / width
        new_height = int(height * scale)
        resized_image = image.resize((target_size, new_height), Image.LANCZOS)
        
        # Create a white square background
        square_image = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        
        # Calculate vertical padding (top and bottom)
        padding_y = (target_size - new_height) // 2
        
        # Paste resized image centered vertically
        square_image.paste(resized_image, (0, padding_y))
        return square_image, (0, padding_y)
    
    else:
        # Portrait image - scale height to target_size
        scale = target_size / height
        new_width = int(width * scale)
        resized_image = image.resize((new_width, target_size), Image.LANCZOS)
        
        # Create a white square background
        square_image = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        
        # Calculate horizontal padding (left and right)
        padding_x = (target_size - new_width) // 2
        
        # Paste resized image centered horizontally
        square_image.paste(resized_image, (padding_x, 0))
        return square_image, (padding_x, 0)


class IMLCropDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        # Read CSV with the new format (name, x, y, crop_shape)
        self.df = pd.read_csv(csv_path)
        
        # Get unique shape classes for index mapping
        # Convert text shapes to integer classes if needed
        if 'crop_shape' in self.df.columns:
            if self.df['crop_shape'].dtype == 'object':
                self.df['shape_class'] = self.df['crop_shape'].apply(
                    lambda x: size_conversion.get_int_from_shape(x) if isinstance(x, str) else int(x)
                )
            else:
                self.df['shape_class'] = self.df['crop_shape']
                
            # Generate the mapping from shape classes to indices (0-based)
            # First, get all unique shape class values from the dataset
            shape_values = self.df['shape_class'].unique()
            
            # Create a list of shape values ordered by the conversion table
            # Only use shapes that exist in our dataset
            ordered_shapes = []
            for shape_id in sorted(size_conversion.INT_TO_SHAPE.keys()):
                if shape_id in shape_values:
                    ordered_shapes.append(shape_id)
            
            # Create the mapping from shape values to indices
            self.shape_to_index = {shape_id: idx for idx, shape_id in enumerate(ordered_shapes)}
            print(f"Shape mapping: {self.shape_to_index}")
        
        # Handle both old and new CSV formats
        if 'crop_shape' in self.df.columns:
            # Format with crop_shape - convert text to int if needed
            self.df['shape_class'] = self.df['crop_shape'].apply(
                lambda x: size_conversion.get_int_from_shape(x) if isinstance(x, str) else int(x)
            )
        elif all(col in self.df.columns for col in ['x', 'y', 'size', 'ratio']):
            # Old format with size and ratio - convert to shape_class
            # This is a placeholder for backward compatibility
            print("Warning: Using old CSV format with size and ratio. This may not be accurate.")
            self.df['shape_class'] = 4  # Default to 256x256 Square (class 4)
        else:
            # Assume unlabeled CSV or custom format
            print("Warning: CSV format not recognized. Assuming columns are: name, x, y, crop_shape")
            self.df.columns = ['name', 'x', 'y', 'crop_shape']
            self.df['shape_class'] = self.df['crop_shape'].apply(
                lambda x: size_conversion.get_int_from_shape(x) if isinstance(x, str) else int(x)
            )
            
        self.img_dir = img_dir
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get image name (handle different column names)
        img_name = row.get('name', row.get('image', f"img_{idx}"))
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img, _ = resize_and_pad_square(img, target_size=224)
        img = transforms.ToTensor()(img)
        img = self.normalize(img)
        
        # Get coordinates and shape class
        coords = torch.tensor([row['x'], row['y']], dtype=torch.float32)  # Just x, y now
        shape_class = int(row['shape_class'])
        
        # Use our shape-to-index mapping to get the correct index
        # This follows the order of the conversion table starting from 0
        shape_class_idx = self.shape_to_index[shape_class]

        return img, (coords, shape_class_idx)
    
def get_loaders(csv_path, img_dir, batch_size=32, val_split=0.25, num_workers=4, seed=42):
    """Create train and validation data loaders.
    
    Args:
        csv_path: Path to CSV file with labels
        img_dir: Directory containing images
        batch_size: Batch size for training
        val_split: Fraction of data to use for validation
        num_workers: Number of workers for data loading
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Set random seed for reproducible splits
    torch.manual_seed(seed)
    
    # Create dataset
    ds = IMLCropDataset(csv_path, img_dir)
    
    # Split into train and validation sets
    val_size = int(len(ds) * val_split)
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader