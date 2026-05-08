"""Compute DINOv2 ViT-S/14 embeddings for sovereign flags."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import timm
from PIL import Image
from timm.data import resolve_data_config, create_transform

ROOT = Path(__file__).resolve().parent.parent
PNG_DIR = ROOT / "data" / "png"
EMB_DIR = ROOT / "data" / "embeddings"
CSV_PATH = ROOT / "data" / "sovereign_flags.csv"

MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
INPUT_SIZE = 518  # DINOv2 native resolution (37 patches * 14)


def letterbox_to_square(img: Image.Image, side: int = INPUT_SIZE) -> Image.Image:
    """Composite RGBA flag onto white square, preserving aspect ratio."""
    img = img.convert("RGBA")
    scale = min(side / img.width, side / img.height)
    new_w = max(1, int(round(img.width * scale)))
    new_h = max(1, int(round(img.height * scale)))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    x = (side - new_w) // 2
    y = (side - new_h) // 2
    canvas.paste(resized, (x, y), resized)
    return canvas


def main() -> None:
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open() as f:
        rows = list(csv.DictReader(f))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    model.eval().to(device)

    cfg = resolve_data_config({}, model=model)
    cfg["input_size"] = (3, INPUT_SIZE, INPUT_SIZE)
    transform = create_transform(**cfg, is_training=False)

    iso2_codes: list[str] = []
    embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for row in rows:
            iso2 = row["iso2"]
            png_path = PNG_DIR / f"{iso2}.png"
            img = letterbox_to_square(Image.open(png_path))
            tensor = transform(img).unsqueeze(0).to(device)
            feats = model.forward_features(tensor)
            if isinstance(feats, dict):
                cls = feats.get("x_norm_clstoken", feats.get("cls_token"))
            else:
                cls = feats[:, 0] if feats.ndim == 3 else feats
            vec = cls.squeeze(0).cpu().numpy().astype(np.float32)
            embeddings.append(vec)
            iso2_codes.append(iso2)

    arr = np.stack(embeddings, axis=0)
    np.save(EMB_DIR / "dinov2_vits14.npy", arr)
    (EMB_DIR / "iso2_order.txt").write_text("\n".join(iso2_codes) + "\n")
    print(f"saved {arr.shape} -> {EMB_DIR / 'dinov2_vits14.npy'}")


if __name__ == "__main__":
    main()
