# dataset.py
import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


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
        # Read CSV and rename columns to match our code's expectations
        self.df = pd.read_csv(csv_path, names=['image', 'x', 'y', 'size', 'ratio'], skiprows=1)
        self.img_dir = img_dir
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row['image'])).convert('RGB')
        
        # Use resize_and_pad_square instead of direct resize
        img, _ = resize_and_pad_square(img, target_size=224)
        
        # Convert to tensor and normalize
        img = transforms.ToTensor()(img)
        img = self.normalize(img)
        
        target = torch.tensor([
            row['x'], row['y'], row['size'], row['ratio']
        ], dtype=torch.float32)
        return img, target


def get_loaders(csv_path, img_dir, batch_size=32, val_split=0.25, num_workers=4):
    ds = IMLCropDataset(csv_path, img_dir)
    val_size = int(len(ds) * val_split)
    train_ds, val_ds = random_split(ds, [len(ds) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader