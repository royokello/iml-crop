import argparse
import os
import shutil
import json
from PIL import Image
import torch
import pandas as pd
from torchvision import transforms
from model import IMLCropModel
from dataset import resize_and_pad_square
from utils import find_latest_stage


def perform_cropping(project: str,
                     stage: int = None,
                     output: str = None,
                     resolution: int = 1024,
                     batch_size: int = 64,
                     verbose: bool = False):
    """
    Crop and save images using the updated classification model with index, x, y, width, and ratio label format.

    Args:
        project: Root project directory
        stage: Stage number (used to form directory name 'stage_{stage}')
        output: Output directory for cropped images (defaults to 'stage_{stage+1}')
        resolution: Output resolution for cropped images
        batch_size: Number of images to process per batch
        verbose: Whether to show detailed logging
    """
    # Determine stage
    if stage is None:
        try:
            stage = find_latest_stage(project)
            print(f"Using latest stage: {stage}")
        except ValueError as e:
            print(f"Error: {e}")
            return

    base = f"stage_{stage}"
    src_path = os.path.join(project, base)
    labels_file = os.path.join(project, f"{base}_crop_labels.csv")
    ratios_file = os.path.join(project, f"{base}_crop_ratios.json")
    model_path = os.path.join(project, f"{base}_crop_model.pth")

    # Set default output
    if output is None:
        output = f"stage_{stage + 1}"
    output_path = os.path.join(project, output)

    # Validate paths
    for path, typ in [(src_path, 'directory'), (labels_file, 'labels file'), (model_path, 'model file'), (ratios_file, 'ratio definitions')]:
        if not os.path.exists(path):
            print(f"Error: {typ} not found at '{path}'")
            return

    os.makedirs(output_path, exist_ok=True)

    # Load labels
    df = pd.read_csv(labels_file)
    required = ['index', 'x', 'y', 'width', 'ratio']
    if any(col not in df.columns for col in required):
        print(f"Error: Labels CSV must contain columns {required}")
        return

    # Load ratio definitions
    with open(ratios_file, 'r') as rf:
        ratio_list = json.load(rf)
    num_ratio_classes = len(ratio_list)
    print(f"Loaded {num_ratio_classes} ratio classes from {ratios_file}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = IMLCropModel(num_ratio_classes=num_ratio_classes).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Prepare image file list
    image_files = sorted([
        f for f in os.listdir(src_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))
    ])
    if not image_files:
        print(f"No images found in '{src_path}'")
        return

    total_images = len(image_files)
    print(f"Found {total_images} images, processing in batches of {batch_size}...")

    # Transform for model input
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Process in batches
    for start_idx in range(0, total_images, batch_size):
        end_idx = min(start_idx + batch_size, total_images)
        batch_names = image_files[start_idx:end_idx]
        batch_images = []
        batch_tensors = []

        for img_name in batch_names:
            img_path = os.path.join(src_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                batch_images.append(img)
                resized, _ = resize_and_pad_square(img, target_size=384)
                batch_tensors.append(transform(resized))
            except Exception as e:
                print(f"Warning: failed to load {img_name}: {e}")

        if not batch_tensors:
            continue

        batch_tensor = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            coords_pred, ratio_logits = model(batch_tensor)
            ratio_indices = torch.argmax(ratio_logits, dim=1).cpu().tolist()

        # Crop and save
        for i, img_name in enumerate(batch_names):
            orig_img = batch_images[i]
            x_norm, y_norm, w_norm = coords_pred[i].cpu().tolist()
            r_idx = ratio_indices[i]
            raw_ratio = ratio_list[r_idx]
            # Parse ratio, which may be a fraction string like '1/1'
            if isinstance(raw_ratio, str) and '/' in raw_ratio:
                num, den = raw_ratio.split('/')
                ratio_val = float(num) / float(den)
            else:
                ratio_val = float(raw_ratio)

            # Calculate pixel values on square canvas
            img_w, img_h = orig_img.size
            square_size = max(img_w, img_h)
            square_img, (pad_x, pad_y) = resize_and_pad_square(orig_img, target_size=square_size)

            x_px = int(x_norm * square_size)
            y_px = int(y_norm * square_size)
            w_px = int(w_norm * square_size)
            h_px = int(w_px / ratio_val)

            # Crop and resize
            crop_box = (x_px, y_px, x_px + w_px, y_px + h_px)
            try:
                cropped = square_img.crop(crop_box)
            except Exception as e:
                print(f"Error cropping {img_name}: {e}")
                continue

            # Scale to resolution
            scale = max(resolution / max(w_px, h_px), 1)
            new_w = int(w_px * scale)
            new_h = int(h_px * scale)
            out_img = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Save
            out_path = os.path.join(output_path, img_name)
            out_img.save(out_path)

            if verbose:
                print(f"Saved {img_name}: crop ({w_px}x{h_px}) -> ({new_w}x{new_h}) at {out_path}")
            else:
                idx_display = start_idx + i + 1
                print(f"[{idx_display}/{total_images}] {img_name} -> {new_w}x{new_h}")

    print(f"Cropping complete. Results in '{output_path}'")


def main():
    parser = argparse.ArgumentParser(description="Batch crop images using IMLCropModel and new label format.")
    parser.add_argument('--project', required=True, help='Root project directory')
    parser.add_argument('--stage', type=int, default=None, help='Stage number (defaults to latest)')
    parser.add_argument('--output', type=str, default=None, help='Output directory name')
    parser.add_argument('--resolution', type=int, default=768, help='Target resolution for crops')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    args = parser.parse_args()
    perform_cropping(
        args.project,
        args.stage,
        args.output,
        args.resolution,
        args.batch_size,
        args.verbose
    )

if __name__ == '__main__':
    main()
