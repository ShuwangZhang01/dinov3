import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(data_dir: Path):
    imgs = []
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
    return sorted(imgs, key=lambda x: str(x))


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def to_uint8(img01: np.ndarray) -> np.ndarray:
    img01 = np.clip(img01, 0.0, 1.0)
    return (img01 * 255.0 + 0.5).astype(np.uint8)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--model", type=str, default="dinov3_convnext_tiny")
    parser.add_argument("--img_size", type=int, default=1080)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    data_dir = (repo_root / args.data_dir).resolve()
    weights_path = (repo_root / args.weights).resolve()
    out_dir = (repo_root / args.output_dir).resolve()
    ensure_dir(out_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")

    # Load model
    import hubconf
    model_fn = getattr(hubconf, args.model)
    model = model_fn(pretrained=True, weights=str(weights_path))
    model.eval().to(args.device)

    # All DINOv3 ConvNeXt models have total stride = 8
    STRIDE = 8
    grid_h = args.img_size // STRIDE
    grid_w = args.img_size // STRIDE

    # Input transform
    tfm = transforms.Compose([
        transforms.Resize(args.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    # Verify output size with dummy input
    with torch.no_grad():
        dummy_img = torch.randn(1, 3, args.img_size, args.img_size).to(args.device)
        feats = model.forward_features(dummy_img)
        patch_tokens = feats["x_norm_patchtokens"]  # (B, H'*W', D)

    B, N, D = patch_tokens.shape
    expected_tokens = grid_h * grid_w
    if N != expected_tokens:
        raise RuntimeError(
            f"Token count mismatch: got {N}, expected {expected_tokens} ({grid_h}x{grid_w}). "
            f"Check image size divisibility by stride={STRIDE}."
        )

    print(f"Feature grid: {grid_h}x{grid_w} = {N} tokens (D={D})")

    # Dataset output path
    dataset_name = Path(args.data_dir).name + "_CNN"
    dataset_out = out_dir / dataset_name
    ensure_dir(dataset_out)

    # Process each image
    paths = list_images(data_dir)
    if not paths:
        raise RuntimeError(f"No images found under: {data_dir}")

    for p in tqdm(paths, desc="Extracting CNN feature maps"):
        # Load original image for overlay
        img_pil = Image.open(p).convert("RGB")
        img_resized = transforms.Resize(args.img_size, interpolation=transforms.InterpolationMode.BICUBIC)(img_pil)
        img_cropped = transforms.CenterCrop(args.img_size)(img_resized)
        img_np = np.asarray(img_cropped).astype(np.float32) / 255.0  # [0,1]

        # Preprocess
        x = tfm(img_pil).unsqueeze(0).to(args.device)

        # Forward pass
        feats = model.forward_features(x)
        patch_tokens = feats["x_norm_patchtokens"]  # (1, H'*W', D)
        patch_tokens = patch_tokens[0].float().cpu().numpy()  # (P, D)

        # Reshape to spatial grid: (D, H', W')
        feat_map = patch_tokens.reshape(grid_h, grid_w, -1).transpose(2, 0, 1)

        # Save feature map (.npy)
        stem = p.stem
        np.save(dataset_out / f"{stem}_feat.npy", feat_map)

        # Optional: PCA visualization to RGB
        fm_flat = feat_map.transpose(1, 2, 0).reshape(-1, feat_map.shape[0])  # (P, D)
        fm3 = PCA(n_components=3, random_state=0).fit_transform(fm_flat)  # (P, 3)
        fm3 = (fm3 - fm3.min(axis=0)) / (np.ptp(fm3, axis=0) + 1e-8)  # normalize
        fm3 = np.clip(fm3, 0, 1)  # ensure valid range
        fm3 = fm3.reshape(grid_h, grid_w, 3)

        # Upsample to original image size
        fm3_up = np.array(
            Image.fromarray(to_uint8(fm3)).resize((args.img_size, args.img_size), resample=Image.BICUBIC)
        )
        Image.fromarray(fm3_up).save(dataset_out / f"{stem}_feat_rgb.png")

    print(f"\n✅ Done. Outputs saved under: {dataset_out}")


if __name__ == "__main__":
    main()
