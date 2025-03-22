import argparse
import os
import shutil
from PIL import Image
import numpy as np
import torch

from utils import get_model_by_latest, get_model_by_name
from predict import predict
from train import find_album_directories
from model import ViTCropper
from dataset import resize_and_pad_square

def main(root_dir: str, resolution: int = 1024, model_name: str|None=None):
    """
    Crop images in src_culled directories and save them to src_cropped directories.
    
    Args:
        root_dir: Root directory containing album directories
        resolution: Resolution of the output images in pixels
        model_name: Optional model name to use for cropping
    """
    # Find all album directories with label data
    album_dirs = find_album_directories(root_dir)
    print(f"Found {len(album_dirs)} album directories with label data")
    
    if len(album_dirs) == 0:
        print("No valid album directories found. Exiting.")
        return
        
    # Look for the model in the root directory
    model_path = os.path.join(root_dir, 'crop_model.pth')
    
    # Load the trained model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Load ViT model
    if os.path.exists(model_path):
        print(f"Loading ViT model from {model_path}")
        model = ViTCropper()
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"No trained model found at {model_path}. Please train the model first.")
        return
    
    model.to(device)
    model.eval()
    
    # Process each album directory
    for album_dir in album_dirs:
        album_name = os.path.basename(album_dir)
        print(f"Processing album: {album_name}")
        
        # Source directory with images to crop
        src_culled_dir = os.path.join(album_dir, 'src_culled')
        
        # Target directory for cropped images
        src_cropped_dir = os.path.join(album_dir, 'src_cropped')
        os.makedirs(src_cropped_dir, exist_ok=True)
        
        # Get list of images in source directory
        image_files = [f for f in os.listdir(src_culled_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        if not image_files:
            print(f"No valid image files found in {src_culled_dir}. Skipping.")
            continue
        
        print(f"Found {len(image_files)} images to process in {album_name}")
        
        # Process each image
        for i, img_file in enumerate(image_files):
            src_path = os.path.join(src_culled_dir, img_file)
            dst_path = os.path.join(src_cropped_dir, img_file)
            
            try:
                # Skip if already processed with the same resolution
                if os.path.exists(dst_path):
                    with Image.open(dst_path) as img:
                        if max(img.size) == resolution:
                            continue
                
                # Get crop predictions in the new format: (x1, y1, height, ratio_code)
                x1, y1, height_norm, ratio_code = predict(device, model, src_path)
                
                # Open the source image
                img = Image.open(src_path)
                width, height_px = img.size
                
                # Determine square dimension as the maximum of width and height.
                square_size = max(width, height_px)
                # Create a square image using the new function (with white background)
                virtual_img, pad = resize_and_pad_square(img, target_size=square_size)
                
                # Calculate pixel coordinates in the square image
                x1_px = int(x1 * square_size)
                y1_px = int(y1 * square_size)
                crop_height_px = int(height_norm * square_size)
                
                # Calculate crop width based on the ratio code
                if ratio_code == 0:  # 1:1 aspect ratio
                    crop_width_px = crop_height_px
                elif ratio_code == 1:  # 2:3 aspect ratio
                    crop_width_px = int(crop_height_px * (2/3))
                elif ratio_code == 2:  # 3:2 aspect ratio
                    crop_width_px = int(crop_height_px * (3/2))
                else:
                    crop_width_px = crop_height_px
                
                # Calculate bottom-right coordinates
                x2_px = x1_px + crop_width_px
                y2_px = y1_px + crop_height_px
                
                # Crop the virtual square image
                cropped_img = virtual_img.crop((x1_px, y1_px, x2_px, y2_px))
                
                # Resize to target resolution, maintaining aspect ratio.
                cropped_width, cropped_height = cropped_img.size
                new_height = resolution
                new_width = int(cropped_width * (resolution / cropped_height))
                resized_img = cropped_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Save the processed image
                resized_img.save(dst_path)
                
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                shutil.copy(src_path, dst_path)
    
    print("Processing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop images using the trained model.")
    parser.add_argument("root_directory", type=str, help="Root directory containing album directories.")
    parser.add_argument("--resolution", "-r", type=int, default=1024, help="Target resolution for output images.")
    parser.add_argument("--model", "-m", type=str, default=None, help="Specific model to use (if multiple are available).")
    
    args = parser.parse_args()
    main(args.root_directory, args.resolution, args.model)
