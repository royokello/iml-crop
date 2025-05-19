# IML Crop

IML Crop is a Python library for intelligent machine learning-based image cropping that uses a Vision Transformer (ViT) model to learn from labeled samples and automatically apply optimal cropping to batches of images.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Modules](#modules)
  - [Label Module](#label-module)
  - [Train Module](#train-module)
  - [Crop Module](#crop-module)

## Installation

```bash
git clone https://github.com/royokello/iml-crop.git
cd iml-crop
pip install -r requirements.txt
```

## Project Structure

The project uses a stage-based workflow structure:

```
project_directory/
├── stage_1/            # First stage images directory
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── stage_1_crop_labels.csv  # Labels for stage 1 images
├── stage_1_crop_model.pth   # Trained model for stage 1
├── stage_1_crop_epoch_log.csv  # Training progress log
├── stage_2/            # Second stage (output from stage 1 cropping)
│   ├── image1.jpg
│   └── ...
└── ...
```

## Modules

### Label Module

The label module provides a web-based interface for manually labeling crop coordinates and shapes in images.

#### Usage

```bash
python label.py --project "path/to/project" [--stage STAGE_NUMBER]
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|--------|
| `--project` | Root project directory containing images | Required |
| `--stage` | Stage number to label (e.g., 1, 2, etc.) | Latest stage number |

**Note:** At least one stage directory (e.g., 'stage_1') must exist in the project directory. The module will raise an error if no stage directories are found.

#### Features

- Interactive web interface (runs on port 5000)
- Visual crop selection with movable boundary box
- Radio button selection for different crop shapes (square, portrait, landscape)
- Navigation between images (previous, next, random)
- Auto-saving of labels to CSV file (`{base}_crop_labels.csv`)
- Statistics display showing distribution of labeled crop shapes

#### CSV Format

The module generates a CSV file with the following columns:
- `name`: Image filename
- `x`, `y`: Normalized coordinates (0-1) of the crop center point
- `crop_shape`: Integer value representing the crop shape

#### How It Works

1. The label interface loads images from the specified directory
2. Users position a crop box on each image and select a crop shape
3. Coordinates are normalized to the canvas size (0.0-1.0 range)
4. Labels are automatically saved to a CSV file as you navigate between images
5. The system tracks statistics on which crop shapes are used most frequently

### Train Module

The train module trains a Vision Transformer model to predict both crop coordinates and shapes based on labeled samples.

#### Usage

```bash
python train.py --project "path/to/project" [--stage STAGE_NUMBER] [options]
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|--------|
| `--project` | Root project directory | Required |
| `--stage` | Stage number to train (e.g., 1, 2, etc.) | Latest stage number |
| `--epochs`, `-e` | Number of training epochs | 50 |
| `--learning_rate`, `-lr` | Learning rate | 1e-4 |
| `--batch_size`, `-b` | Batch size | 16 |
| `--val_split`, `-v` | Validation split ratio | 0.2 |
| `--patience`, `-p` | Early stopping patience in epochs | 10 |
| `--model`, `-m` | HuggingFace ViT model name | 'google/vit-base-patch16-224' |

**Note:** At least one stage directory (e.g., 'stage_1') must exist in the project directory. The module will raise an error if no stage directories are found.

#### Features

- Dual-head model architecture for predicting both coordinates and shape class
- Separate tracking of coordinate and shape losses
- CSV logging with per-epoch metrics for both training and validation
- Model checkpoint saving based on improvement in both coordinate and shape losses
- Early stopping to prevent overfitting
- Learning rate scheduling with automatic reduction on plateau

#### Training Process

1. Reads labels from `{base}_crop_labels.csv` file in the specified stage directory
2. Automatically determines the shape classes from the dataset
3. Splits data into training and validation sets based on the specified ratio
4. Trains the ViT model with dual prediction heads (coordinates and shape)
5. Logs progress to CSV file with columns:
   - epoch
   - tr_coord_loss (training coordinate loss)
   - tr_shape_loss (training shape classification loss)
   - val_coord_loss (validation coordinate loss)
   - val_shape_loss (validation shape classification loss)
6. Saves best model as `{base}_crop_model.pth` when BOTH coordinate and shape losses improve
7. Uses early stopping when no improvement is seen after the specified patience epochs

### Crop Module

The crop module uses a trained model to automatically process and crop images in a specified stage directory.

#### Usage

```bash
python crop.py --project "path/to/project" --stage 1 [options]
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|--------|
| `--project` | Root project directory | Required |
| `--stage` | Stage number to process (forms directory name 'stage_{stage}') | Required |
| `--output` | Output directory for cropped images | 'stage_{stage+1}' |
| `--resolution` | Target resolution for output images | 768 |
| `--verbose` | Enable detailed per-image logging | False |

#### Features

- Automatic batch processing of all images in a stage directory
- Dynamic output directory naming based on stage progression
- Shape-aware cropping that respects the predicted crop shape
- High-resolution output with configurable target resolution
- Detailed logging option for troubleshooting

#### Cropping Process

1. Loads trained model from `stage_{stage}_crop_model.pth`
2. Determines the shape mapping from the `stage_{stage}_crop_labels.csv` file
3. For each image in the stage directory:
   - Predicts center coordinates (x, y) and shape class
   - Calculates crop dimensions based on the predicted shape class
   - Creates a square padded version of the image for consistent cropping
   - Crops the image at the predicted coordinates with appropriate dimensions
   - Resizes to the target resolution maintaining aspect ratio
   - Saves to the output directory (`stage_{stage+1}` by default)

#### Shape Classes

The system supports multiple crop shapes mapped as integer values:
- Square formats (1:1 ratio): 1-4
- Portrait formats (2:3 ratio): 5-10
- Landscape formats (3:2 ratio): 11-12

Each shape class corresponds to specific pixel dimensions when displayed at 512px canvas size.

## Workflow Example

1. **Label**: Label images in stage_1 directory
   ```bash
   python label.py --project "project_dir" --stage 1
   ```

2. **Train**: Train a model using the labeled data
   ```bash
   python train.py --project "project_dir" --stage 1 --epochs 50
   ```

3. **Crop**: Apply the trained model to crop all images
   ```bash
   python crop.py --project "project_dir" --stage 1
   ```
   This will create a stage_2 directory with the cropped results.

4. **Repeat**: For multi-stage processing, repeat the workflow with the next stage
   ```bash
   python label.py --project "project_dir" --stage 2
   python train.py --project "project_dir" --stage 2 --epochs 50
   python crop.py --project "project_dir" --stage 2
   ```
   
5. **Automatic Stage Detection**: You can also omit the stage parameter to automatically use the latest stage
   ```bash
   python label.py --project "project_dir"
   python train.py --project "project_dir" --epochs 50
   python crop.py --project "project_dir"
   ```
