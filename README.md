# ML Cropper
ml-cropper is a Python library for training an image cropping model with a few samples and batch cropping images in a directory.

# Installation
```
git clone https://github.com/royokello/ml-cropper.git
cd ml-cropper
pip install -r requirements.txt
```

# Usage

## Data Structure

The project now uses the following directory structure:

```
album_directory
├── src_culled      # Directory containing all source images to label
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── crop_labels.csv # CSV file containing crop labels (img_name, x1, y1, height, ratio)
```

The label format has been updated to use:
- `x1, y1`: coordinates of the top-left corner (normalized to canvas size)
- `height`: height of the crop box (normalized to canvas size)
- `ratio`: aspect ratio code (0 for 1:1, 1 for 2:3, 2 for 3:2)

## Convert Labels

If you have existing labels in the old format (x1, y1, x2, y2), you can convert them to the new format:

```
python convert_labels.py "path to root directory" [--dry-run]
```

The convert_labels utility:
- Automatically detects and converts old format labels to the new format
- Works on a single album directory or a root directory with multiple albums
- Use the `--dry-run` flag to preview changes without modifying files

## Label

```
python label.py "path to album directory"
```

The label module opens a web interface for labeling images. The images are loaded from the `src_culled` directory within your album directory, and labels are saved to `crop_labels.csv` in the album directory.

## Train

```
python train.py "path to root directory" [options]
```

The train module now accepts a root directory containing multiple album directories. It automatically:
1. Scans for all album directories containing `crop_labels.csv` files and `src_culled` image directories
2. Creates a combined dataset from all valid albums
3. Trains a model and saves it as `crop_model.pth` in the root directory
4. Tracks training progress across runs using `crop_epoch.txt` and `crop_val_loss.txt`

### Options:
- `-e, --epochs` : Number of epochs to train (default: 256)
- `-l, --learning_rate` : Learning rate (default: 0.001)
- `-b, --batch_size` : Batch size (default: 64)
- `-v, --val_split` : Validation split ratio (default: 0.2)
- `-p, --patience` : Early stopping patience in epochs (default: 8)

## Crop

```
python crop.py -w "path to album directory"
