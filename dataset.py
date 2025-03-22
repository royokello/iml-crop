import os
import json
import torch
import csv
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F


from PIL import Image

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