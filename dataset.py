import os
import json
import torch
import csv
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F


def resize_and_pad_square(image, target_size=224):
    """
    Resize an image to a square while maintaining aspect ratio and padding.
    
    Args:
        image: PIL Image
        target_size: Target height (and width) for the square output
        
    Returns:
        Resized and padded PIL Image as a square
    """
    # Get original dimensions
    width, height = image.size
    
    # Determine scaling factor to match height to target_size
    scale = target_size / height
    new_width = int(width * scale)
    
    # Resize image to match target height while preserving aspect ratio
    resized_image = image.resize((new_width, target_size), Image.LANCZOS)
    
    # Create a square blank image with padding
    square_image = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    
    # Calculate horizontal padding
    padding_x = (target_size - new_width) // 2
    
    # Paste the resized image onto the square image
    square_image.paste(resized_image, (padding_x, 0))
    
    return square_image, padding_x


class ImageDataset(Dataset):
    def __init__(self, low_res_dir, labels_file, transform=None):
        self.low_res_dir = low_res_dir
        self.transform = transform
        self.labels = {}
        
        # Read labels from CSV file
        if os.path.exists(labels_file):
            with open(labels_file, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header row
                for row in reader:
                    if len(row) == 5:  # Ensure row has 5 columns (img_name, x1, y1, height, ratio)
                        img_name = row[0]
                        # Parse the new format: x1, y1, height, ratio
                        coords = [float(row[1]), float(row[2]), float(row[3]), int(row[4])]
                        self.labels[img_name] = coords
                        
        # Filter out images that don't exist in the directory
        self.image_names = []
        for img_name in self.labels.keys():
            img_path = os.path.join(self.low_res_dir, img_name)
            if os.path.exists(img_path):
                self.image_names.append(img_name)
            
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.low_res_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # Get original dimensions
        width, height = image.size
        
        # Resize and pad to square format (same as during inference)
        target_size = 224
        
        # Determine scaling factor to match height to target_size
        scale = target_size / height
        new_width = int(width * scale)
        
        # Resize image to match target height while preserving aspect ratio
        resized_image = image.resize((new_width, target_size), Image.LANCZOS)
        
        # Create a square blank image with padding
        square_image = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        
        # Calculate horizontal padding
        padding_x = (target_size - new_width) // 2
        
        # Paste the resized image onto the square image
        square_image.paste(resized_image, (padding_x, 0))
        
        # Apply any additional transform if needed
        if self.transform:
            square_image = self.transform(square_image)
        
        # Get bounding box coordinates in the new format: x1, y1, height, ratio
        bbox = self.labels[img_name]
        bbox = torch.tensor(bbox, dtype=torch.float32)

        return square_image, bbox