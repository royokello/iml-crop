import torch
from PIL import Image
import numpy as np


def resize_and_pad_square(image, target_size=224):
    """
    Resize an image to a square while maintaining aspect ratio and padding.
    
    Args:
        image: PIL Image
        target_size: Target height (and width) for the square output
        
    Returns:
        square_image: Resized and padded PIL Image as a square
        padding_x: Amount of horizontal padding added
    """
    # Get original dimensions
    width, height = image.size
    
    # Determine scaling factor to match height to target_size
    scale = target_size / height
    new_width = int(width * scale)
    
    # Resize image to match target height while preserving aspect ratio
    resized_image = image.resize((new_width, target_size), Image.Resampling.LANCZOS)
    
    # Create a square blank image with padding
    square_image = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    
    # Calculate horizontal padding
    padding_x = (target_size - new_width) // 2
    
    # Paste the resized image onto the square image
    square_image.paste(resized_image, (padding_x, 0))
    
    return square_image, padding_x


def predict(device: torch.device, model, image_path: str) -> tuple[float, float, float, int]:
    """
    Predicts the coordinates of the bounding box for an object in the given image using the ViT model.
    
    Args:
        device: The torch device to use
        model: The ViT model to use for prediction
        image_path: Path to the image file
    
    Returns:
        tuple (x1, y1, height, ratio) where:
        - x1, y1: coordinates of the top-left corner, normalized to canvas size
        - height: height of the crop box, normalized to canvas size
        - ratio: aspect ratio code (0 for 1:1, 1 for 2:3, 2 for 3:2)
    """
    # Load the image
    original_image = Image.open(image_path).convert('RGB')
    original_width, original_height = original_image.size
    
    # Resize and pad the image to a square
    image, padding_x = resize_and_pad_square(original_image, target_size=224)
    
    # Put model in evaluation mode
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        # Forward pass through the model
        # Model now returns the combined tensor [x1, y1, height, ratio] in inference mode
        outputs = model([image])
        
        # Get the predicted values
        predicted_values = outputs.squeeze().cpu().numpy()
        
        # Extract components
        x1, y1, height, ratio_code = predicted_values
        
        # Ensure ratio_code is an integer (0, 1, or 2)
        ratio_code = int(round(ratio_code))
        
        # Clamp values within valid ranges
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        height = max(0.1, min(1.0, height))  # Prevent zero or negative height
        ratio_code = max(0, min(2, ratio_code))  # Ensure ratio code is 0, 1, or 2
    
    return x1, y1, height, ratio_code
