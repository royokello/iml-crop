import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


def resize_and_pad_square(image, target_size=384):
    """
    Resize an image to a square while maintaining aspect ratio and padding.

    Returns:
        square_image (PIL.Image), padding (tuple)
    """
    width, height = image.size
    if width >= height:
        scale = target_size / width
        new_height = int(height * scale)
        resized = image.resize((target_size, new_height), Image.LANCZOS)
        square = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        padding_y = (target_size - new_height) // 2
        square.paste(resized, (0, padding_y))
        return square, (0, padding_y)
    else:
        scale = target_size / height
        new_width = int(width * scale)
        resized = image.resize((new_width, target_size), Image.LANCZOS)
        square = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        padding_x = (target_size - new_width) // 2
        square.paste(resized, (padding_x, 0))
        return square, (padding_x, 0)


class IMLCropDataset(Dataset):
    def __init__(self, samples_path: str, images_path: str):
        self.df = pd.read_csv(samples_path)
        self.img_dir = images_path

        # List and sort all image files once
        self.image_files = sorted(
            f for f in os.listdir(images_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
        )

        # Verify required columns
        required = ['index', 'x', 'y', 'width', 'ratio']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in {samples_path}, expected {required}")

        # Cast dtypes for numeric stability
        self.df['index'] = self.df['index'].astype(int)
        self.df['x'] = self.df['x'].astype(float)
        self.df['y'] = self.df['y'].astype(float)
        self.df['width'] = self.df['width'].astype(float)
        self.df['ratio'] = self.df['ratio'].astype(int)

        # Image normalization (ImageNet stats)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # Ensure we use a Python int, not numpy type
        img_idx = int(row['index'])
        if img_idx < 0 or img_idx >= len(self.image_files):
            raise IndexError(
                f"Image index {img_idx} out of range (0 to {len(self.image_files)-1})"
            )
        img_name = self.image_files[img_idx]
        img_path = os.path.join(self.img_dir, img_name)

        img = Image.open(img_path).convert('RGB')
        img, _ = resize_and_pad_square(img)
        tensor = transforms.ToTensor()(img)
        img = self.normalize(tensor)

        coords = torch.tensor([row['x'], row['y'], row['width']], dtype=torch.float32)
        ratio_class = int(row['ratio'])
        return img, (coords, ratio_class)


def get_loaders(
    samples_path: str,
    images_path: str,
    batch_size: int = 8,
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42
):
    torch.manual_seed(seed)
    dataset = IMLCropDataset(samples_path, images_path)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader
