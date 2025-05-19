import argparse
import os
import shutil
from PIL import Image
import torch
from torchvision import transforms
from model import create_model
from dataset import resize_and_pad_square

def predict(device, model, image_path):
    """
    Predict crop coordinates for an image using the trained model.
    
    Args:
        device: The device to run inference on (cuda or cpu)
        model: The trained CNN model
        image_path: Path to the image file
    
    Returns:
        tuple: (x1, y1, height, ratio) normalized coordinates
    """
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    
    # Use resize_and_pad_square to match training preprocessing
    img, _ = resize_and_pad_square(img, target_size=224)
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        x1, y1, height, ratio = output[0].cpu().numpy()
    
    return x1, y1, height, ratio

def perform_cropping(input_dir: str, source_dir: str = 'src', resolution: int = 1024):
    """
    Crop images from source directory and save them in a crop directory.
    
    Args:
        input_dir: Root directory containing the source folder with images
        source_dir: Name of the source directory containing images (default: src)
        resolution: Resolution of the output images in pixels
    """
    # Setup paths
    src_path = os.path.join(input_dir, source_dir)
    crop_path = os.path.join(input_dir, 'crop')
    model_path = os.path.join(input_dir, 'crop_model.pth')
    
    # Reset crop directory (delete if exists, then recreate)
    if os.path.exists(crop_path):
        shutil.rmtree(crop_path)
    os.makedirs(crop_path, exist_ok=True)
    
    # Check if source directory exists
    if not os.path.exists(src_path):
        print(f"Error: Source directory {src_path} not found")
        return
        
    # Load the trained model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}. Please train the model first.")
        return
        
    # Load and setup model
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Get list of images to process
    image_files = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    if not image_files:
        print(f"No valid image files found in {src_path}")
        return
        
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for i, img_file in enumerate(image_files):
        src_img_path = os.path.join(src_path, img_file)
        dst_img_path = os.path.join(crop_path, img_file)
        
        try:
            # Skip if already processed with the same resolution
            if os.path.exists(dst_img_path):
                with Image.open(dst_img_path) as img:
                    if max(img.size) == resolution:
                        continue
            
            # Get crop predictions
            x1, y1, height_norm, ratio = predict(device, model, src_img_path)
            print(f" * {x1} {y1} {height_norm} {ratio}")
            
            # Open and process the source image
            img = Image.open(src_img_path)
            width, height_px = img.size
            
            # Create a square canvas with padding
            square_size = max(width, height_px)
            virtual_img, pad = resize_and_pad_square(img, target_size=square_size)
            
            # Calculate pixel coordinates in the square image
            x1_px = int(x1 * square_size)
            y1_px = int(y1 * square_size)
            crop_height_px = int(height_norm * square_size)
            
            # Calculate crop width based on ratio
            # ratio = (width/height)/2, so width = height * ratio * 2
            crop_width_px = int(crop_height_px * ratio * 2)
            
            # Calculate bottom-right coordinates
            x2_px = x1_px + crop_width_px
            y2_px = y1_px + crop_height_px
            
            # Crop the virtual square image
            cropped_img = virtual_img.crop((x1_px, y1_px, x2_px, y2_px))
            
            # Resize to target resolution while maintaining aspect ratio
            cropped_width, cropped_height = cropped_img.size
            new_height = resolution
            new_width = int(cropped_width * (resolution / cropped_height))
            resized_img = cropped_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save the processed image
            resized_img.save(dst_img_path)
            print(f"Processed {i+1}/{len(image_files)}: {img_file}")
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            # If error occurs, copy original image as fallback
            shutil.copy(src_img_path, dst_img_path)
    
    print("Processing completed.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Root directory containing the source folder with images')
    p.add_argument('--source', default='src', help='Name of the source directory containing images (default: src)')
    p.add_argument('--resolution', type=int, default=1024, help='Target resolution for output images')
    args = p.parse_args()
    
    perform_cropping(args.input, args.source, args.resolution)

if __name__ == '__main__':
    main()