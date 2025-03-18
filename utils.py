import json
import logging
import os
import time
import torch
import csv

from model import ViTCropper

def setup_logging(working_dir):
    log_file_path = os.path.join(working_dir, 'training.log')

    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
def generate_model_name(base_model: str | None, samples: int, epochs: int) -> str:
    """
    Generate a unique model name based on current timestamp, base model (if any), number of samples, and epochs.
    """
    result = f"{int(time.time())}"
    if base_model:
        result += f"_b={base_model}"
    
    result += f"_s={samples}_e={epochs}"
    
    return result

def get_model_by_name(device: torch.device, directory: str, name: str) -> torch.nn.Module:
    """
    
    """
    
    model = ViTCropper()  # Initialize your model architecture

    for file in os.listdir(directory):
        if file.startswith(name):
            model_path = os.path.join(directory, file)
            break
    else:
        raise ValueError(f"No model starting with {name} found in {directory}")

    model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)
    
    return model

def get_model_by_latest(device: torch.device, directory: str|None=None) -> torch.nn.Module:
    """
    Load a model whose model name is the latest time from the specified directory and move it to the specified device.
    """
    model = ViTCropper()

    if directory and os.path.exists(directory):
        model_files = [f for f in os.listdir(directory) if f.endswith('.pth')]
        if not model_files:
            raise ValueError(f"No model files found in {directory}")

        latest_model = max(model_files, key=lambda x: int(x.split('_')[0]))
        print(f"latest model: {latest_model}")
        
        model_path = os.path.join(directory, latest_model)

        model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)
    
    return model

def get_labels(directory: str) -> dict[str, list[float]]:
    """
    Load labels from a CSV file in the specified directory.
    Format: img_name, x1, y1, height, ratio
    where x1, y1, height are normalized to canvas size
    ratio values: 0 for 1:1, 1 for 2:3, 2 for 3:2
    """
    labels = {}
    labels_file = os.path.join(directory, 'crop_labels.csv')
    if os.path.exists(labels_file):
        with open(labels_file, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header row
            for row in reader:
                if len(row) == 5:  # Ensure row has 5 columns (img_name, x1, y1, height, ratio)
                    img_name = row[0]
                    coords = [float(row[1]), float(row[2]), float(row[3]), int(row[4])]
                    labels[img_name] = coords
    return labels

def save_labels(directory: str, labels: dict[str, list[float]]):
    """
    Save labels to a CSV file in the specified directory.
    Format: img_name, x1, y1, height, ratio
    where x1, y1, height are normalized to canvas size
    ratio values: 0 for 1:1, 1 for 2:3, 2 for 3:2
    """
    labels_file = os.path.join(directory, 'crop_labels.csv')
    os.makedirs(os.path.dirname(labels_file), exist_ok=True)
    
    with open(labels_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['img_name', 'x1', 'y1', 'height', 'ratio'])  # Updated header
        for img_name, coords in labels.items():
            writer.writerow([img_name, coords[0], coords[1], coords[2], int(coords[3])])

def log_print(message):
    print(message)
    logging.info(message)