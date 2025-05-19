import argparse
import os
import shutil
import json
from PIL import Image
import torch
import pandas as pd
from torchvision import transforms
from model import ViTCropper
from dataset import resize_and_pad_square
import size_conversion

def get_shape_mapping(labels_file):
    """
    Get the mapping from shape class indices to actual shape values.
    """
    # Read the CSV file to determine the unique shape classes
    df = pd.read_csv(labels_file)
    
    # Get shape classes
    if 'crop_shape' in df.columns:
        if df['crop_shape'].dtype == 'object':
            # Convert text shapes to integer classes
            shape_ints = df['crop_shape'].apply(
                lambda x: size_conversion.get_int_from_shape(x) if isinstance(x, str) else int(x)
            )
        else:
            # Already integers
            shape_ints = df['crop_shape']
        
        # Create ordered shape classes based on conversion table
        shape_values = shape_ints.unique()
        ordered_shapes = []
        
        # Add shapes in order of the conversion table
        for shape_id in sorted(size_conversion.INT_TO_SHAPE.keys()):
            if shape_id in shape_values:
                ordered_shapes.append(shape_id)
        
        # Create mapping from indices to shape classes
        index_to_shape = {idx: shape_id for idx, shape_id in enumerate(ordered_shapes)}
        print(f"Shape mapping: {index_to_shape}")
        return index_to_shape
    else:
        raise ValueError("CSV file does not contain 'crop_shape' column")


def predict(device, model, image_path, index_to_shape):
    """
    Predict crop coordinates and shape using the ViTCropper.
    Returns normalized (x, y) coordinates and shape class.
    """
    img = Image.open(image_path).convert('RGB')
    img, _ = resize_and_pad_square(img, target_size=224)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        # Get coordinates and shape logits from the model
        coords, shape_logits = model(tensor)
        x, y = coords[0].cpu().tolist()
        
        # Select shape class
        shape_idx = int(torch.argmax(shape_logits, dim=1)[0].cpu())
        shape_class = index_to_shape[shape_idx]  # Convert index to actual shape class
        
        # Get dimensions for this shape
        width, height = size_conversion.get_dimensions_from_int(shape_class)
        
    return x, y, shape_class, width, height


def perform_cropping(project: str,
                     stage: int,
                     output: str = None,
                     resolution: int = 1024,
                     verbose: bool = False):
    """
    Crop and save images using the updated classification model.
    
    Args:
        project: Root project directory
        stage: Stage number (used to form directory name 'stage_{stage}')
        output: Output directory for cropped images (defaults to 'stage_{stage+1}')
        resolution: Output resolution for cropped images
        verbose: Whether to show detailed logging
    """
    # Form the base name from stage number
    base = f"stage_{stage}"
    
    # Set up paths
    src_path = os.path.join(project, base)
    labels_file = os.path.join(project, f'{base}_crop_labels.csv')
    model_path = os.path.join(project, f'{base}_crop_model.pth')
    
    # Default output directory if not specified
    if output is None:
        # Use the next stage number for output
        next_stage = stage + 1
        output = f"stage_{next_stage}"
    output_path = os.path.join(project, output)

    # Verify directories and files
    if not os.path.isdir(src_path):
        print(f"Error: Source directory '{src_path}' not found.")
        return
    
    if not os.path.isfile(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    # Create output directory
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")

    try:
        # Get shape mapping from labels file
        index_to_shape = get_shape_mapping(labels_file)
        num_shape_classes = len(index_to_shape)
        
        # Initialize model with shape classes
        model = ViTCropper(pretrained_model_name='google/vit-base-patch16-224',
                           num_shape_classes=num_shape_classes).to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        
        # Get list of image files
        files = sorted([f for f in os.listdir(src_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])
        if not files:
            print(f"No images in '{src_path}'.")
            return
    
        print(f"Found {len(files)} images to process.")
        for idx, fname in enumerate(files, 1):
            if verbose:
                print(f"--- Image {idx}/{len(files)}: {fname} ---")
            src_file = os.path.join(src_path, fname)
            dst_file = os.path.join(output_path, fname)
            try:
                # Predict coordinates and shape
                x, y, shape_class, width, height = predict(device, model, src_file, index_to_shape)
                
                if verbose:
                    shape_name = size_conversion.get_shape_from_int(shape_class)
                    print(f"Prediction: x={x:.4f}, y={y:.4f}, shape={shape_name} ({width}x{height})")
    
                # Open and process the image
                img = Image.open(src_file)
                img_w, img_h = img.size
                
                # Create square image for consistent cropping
                square, (padding_x, _) = resize_and_pad_square(img, target_size=max(img_w, img_h))
                square_size = max(img_w, img_h)
    
                # Calculate crop coordinates in pixels
                x_px = int(x * square_size) 
                y_px = int(y * square_size)
                
                # Use the dimensions from the predicted shape
                w_px = (width / 512) * square_size
                h_px = (height / 512) * square_size
                
                if verbose:
                    print(f"Crop box: x={x_px}, y={y_px}, width={w_px}, height={h_px}")
    
                # Crop and resize the image
                cropped = square.crop((x_px, y_px, x_px + w_px, y_px + h_px))
                cw, ch = cropped.size
                
                # Resize to target resolution maintaining aspect ratio
                if height > width:  # Portrait
                    new_h = resolution
                    new_w = int(cw * new_h / ch)
                else:  # Landscape or square
                    new_w = resolution
                    new_h = int(ch * new_w / cw)
                    
                out = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
                out.save(dst_file)
    
                if not verbose:
                    print(f"[{idx}/{len(files)}] Cropped '{fname}' -> {w_px}x{h_px}")
            except Exception as e:
                print(f"Error processing '{fname}': {e}")
                # Copy original file if cropping fails
                shutil.copy(src_file, dst_file)
        print(f"All images processed. Cropped images saved to '{output_path}'")
    except Exception as e:
        print(f"Error during cropping process: {e}")


def main():
    parser = argparse.ArgumentParser(description="Batch crop images using ViTCropper model.")
    parser.add_argument('--project', required=True,
                        help='Root project directory containing data and labels')
    parser.add_argument('--stage', type=int, required=True,
                        help='Stage number (used to form directory name "stage_{stage}")')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for cropped images (defaults to stage_{stage+1})')
    parser.add_argument('--resolution', type=int, default=768,
                        help='Target resolution for output images')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable detailed per-image logging')
    args = parser.parse_args()
    perform_cropping(
        args.project, 
        args.stage, 
        args.output,
        args.resolution, 
        verbose=args.verbose
    )

if __name__ == '__main__':
    main()
