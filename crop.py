import argparse
import os
import random
import json
from pathlib import Path
from typing import List

from PIL import Image
import torch
import pandas as pd
from torchvision import transforms

from model import IMLCropModel
from dataset import resize_and_pad_square
from utils import find_latest_stage


# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

def best_epoch_from_log(log_path: str) -> int:
    """Return the epoch that has the smallest absolute loss_gap."""
    best_gap = float("inf")
    best_epoch = None
    with open(log_path, "r") as f:
        next(f)  # skip header
        for line in f:
            if not line.strip():
                continue
            epoch, _tr, _val, gap = line.split(",")
            gap = abs(float(gap))
            if gap < best_gap:
                best_gap = gap
                best_epoch = int(epoch)
    if best_epoch is None:
        raise RuntimeError("No valid rows in epoch log – cannot pick best epoch.")
    return best_epoch


# ────────────────────────────────────────────────────────────────────────────────
# Core
# ────────────────────────────────────────────────────────────────────────────────

def perform_cropping(project: str,
                     stage: int | None = None,
                     output: str | None = None,
                     resolution: int = 768,
                     batch_size: int = 128,
                     verbose: bool = False,
                     test: bool = False,
                     samples: int = 0,
                     epoch: int | None = None):
    """Crop images (or run test predictions) using stored epoch models.

    Parameters
    ----------
    project : str
        Root project directory.
    stage : int | None
        Stage number (defaults to latest if None).
    output : str | None
        Output directory (ignored in --test; created automatically).
    resolution : int
        Longest side of saved crop.
    batch_size : int
        Batch size for inference.
    verbose : bool
        Verbose logging.
    test : bool
        If True, run test mode (save crops for *all* epoch models).
    samples : int
        When --test, randomly pick this many images (0 = all).
    epoch : int | None
        When not --test, crop with this epoch_XX model; if None pick best by
        smallest loss_gap in the epoch log.
    """

    # ───── determine stage ────────────────────────────────────────────────────
    if stage is None:
        stage = find_latest_stage(project)
        print(f"Using latest stage: {stage}")
    base = f"stage_{stage}"

    src_path = os.path.join(project, base)
    labels_file = os.path.join(project, f"{base}_crop_labels.csv")
    ratios_file = os.path.join(project, f"{base}_crop_ratios.json")

    # model storage dir and epoch log
    models_dir = os.path.join(project, f"{base}_crop_models")
    epoch_log_file = os.path.join(models_dir, f"{base}_crop_epoch_log.csv")

    # validate essential paths
    for path, desc in (
            (src_path, "image directory"),
            (labels_file, "labels csv"),
            (ratios_file, "ratio json"),
            (models_dir, "models directory")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{desc} not found: {path}")

    # gather models
    model_files: List[Path] = sorted(Path(models_dir).glob("epoch_*.pth"),
                                     key=lambda p: int(p.stem.split("_")[1]))
    if not model_files:
        raise RuntimeError("No epoch_*.pth models found in models dir.")

    if test:
        chosen_models = model_files
    else:
        if epoch is not None:
            m = Path(models_dir) / f"epoch_{epoch}.pth"
            if not m.exists():
                raise FileNotFoundError(m)
            chosen_models = [m]
        else:
            best_ep = best_epoch_from_log(epoch_log_file)
            m = Path(models_dir) / f"epoch_{best_ep}.pth"
            chosen_models = [m]
            print(f"[auto] selected epoch {best_ep} (smallest loss_gap)")

    # ───── prepare output dirs ────────────────────────────────────────────────
    if test:
        test_out_root = os.path.join(project, f"{base}_test")
        os.makedirs(test_out_root, exist_ok=True)
        out_root = test_out_root
    else:
        if output is None:
            output = f"stage_{stage + 1}"
        out_root = os.path.join(project, output)
        os.makedirs(out_root, exist_ok=True)

    # ───── load labels & ratios ───────────────────────────────────────────────
    df = pd.read_csv(labels_file)
    required_cols = ["index", "x", "y", "width", "ratio"]
    if any(c not in df.columns for c in required_cols):
        raise ValueError(f"Labels csv must include {required_cols}")

    with open(ratios_file, "r") as f:
        ratio_list = json.load(f)
    num_ratio_classes = len(ratio_list)

    # ───── choose test subset if needed ───────────────────────────────────────
    image_files = sorted([
        f for f in os.listdir(src_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))
    ])
    if not image_files:
        raise RuntimeError("No images found to crop.")

    if test and samples > 0 and samples < len(image_files):
        random.seed(42)
        image_files = random.sample(image_files, samples)

    # ───── common transforms ─────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ───── iterate over selected models ───────────────────────────────────────
    for model_path in chosen_models:
        epoch_tag = model_path.stem.split("_")[1]
        print(f"\n▶ Cropping with model {model_path.name} …")

        model = IMLCropModel(num_ratio_classes=num_ratio_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # batch inference
        total_imgs = len(image_files)
        for start in range(0, total_imgs, batch_size):
            names_batch = image_files[start:start + batch_size]
            pil_batch, tensor_batch = [], []
            for n in names_batch:
                img = Image.open(os.path.join(src_path, n)).convert("RGB")
                pil_batch.append(img)
                resized, _ = resize_and_pad_square(img, target_size=384)
                tensor_batch.append(transform(resized))
            if not tensor_batch:
                continue
            batch_tensor = torch.stack(tensor_batch).to(device)
            with torch.no_grad():
                coords_pred, ratio_logits = model(batch_tensor)
                ratios_idx = torch.argmax(ratio_logits, dim=1).cpu().tolist()

            # loop single images
            for i, name in enumerate(names_batch):
                x_norm, y_norm, w_norm = coords_pred[i].cpu().tolist()
                r_idx = ratios_idx[i]
                raw_r = ratio_list[r_idx]
                r_val = (float(raw_r.split('/')[0]) / float(raw_r.split('/')[1])
                         if isinstance(raw_r, str) and '/' in raw_r else float(raw_r))

                img_w, img_h = pil_batch[i].size
                square_size = max(img_w, img_h)
                square_img, _ = resize_and_pad_square(pil_batch[i], target_size=square_size)

                x_px = int(x_norm * square_size)
                y_px = int(y_norm * square_size)
                w_px = int(w_norm * square_size)
                h_px = int(w_px / r_val)

                crop_box = (x_px, y_px, x_px + w_px, y_px + h_px)
                try:
                    crop = square_img.crop(crop_box)
                except Exception as e:
                    print(f"Error cropping {name}: {e}")
                    continue

                scale = max(resolution / max(w_px, h_px), 1)
                new_w, new_h = int(w_px * scale), int(h_px * scale)
                out_img = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)

                # build output path
                if test:
                    save_name = f"{Path(name).stem}_epoch_{epoch_tag}.png"
                else:
                    save_name = name
                out_path = os.path.join(out_root, save_name)
                out_img.save(out_path)

                if verbose:
                    print(f"  saved {save_name} ({new_w}×{new_h}) -> {out_path}")

        print(f"Finished model {model_path.name}")


# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("Batch crop / test IMLCropModel")
    parser.add_argument('--project', required=True)
    parser.add_argument('--stage', type=int)
    parser.add_argument('--output')
    parser.add_argument('--resolution', type=int, default=768)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--samples', type=int, default=0,
                        help='Random samples for --test')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--epoch', type=int,
                        help='Use this epoch_XX.pth (ignored in --test)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    perform_cropping(project=args.project,
                     stage=args.stage,
                     output=args.output,
                     resolution=args.resolution,
                     batch_size=args.batch_size,
                     verbose=args.verbose,
                     test=args.test,
                     samples=args.samples,
                     epoch=args.epoch)


if __name__ == '__main__':
    main()
