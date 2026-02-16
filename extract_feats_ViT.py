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
    parser.add_argument("--model", type=str, default="dinov3_vits16")
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--layer", type=int, default=-1, help="Which block to use for attention (default: last block).")
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

    # Load DINOv3 model
    import hubconf  # repo hubconf.py
    model_fn = getattr(hubconf, args.model)
    model = model_fn(pretrained=True, weights=str(weights_path))
    model.eval().to(args.device)

    patch = model.patch_size
    grid_h = args.img_size // patch
    grid_w = args.img_size // patch

    # Preprocess (same resize/crop as your embedding script)
    tfm = transforms.Compose([
        transforms.Resize(args.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    # --- Hook to capture qkv from chosen block for attention computation ---
    qkv_cache = {}

    # pick block index
    blk_idx = args.layer if args.layer >= 0 else (len(model.blocks) + args.layer)
    blk_idx = int(blk_idx)
    if blk_idx < 0 or blk_idx >= len(model.blocks):
        raise ValueError(f"layer index out of range: {args.layer} (resolved to {blk_idx})")

    def qkv_hook(module, inp, out):
        # out shape: (B, N, 3*C)
        qkv_cache["qkv"] = out.detach()

    hook_handle = model.blocks[blk_idx].attn.qkv.register_forward_hook(qkv_hook)

    # Process images
    paths = list_images(data_dir)
    if not paths:
        raise RuntimeError(f"No images found under: {data_dir}")

    dataset_name = Path(args.data_dir).name + "_ViT"
    dataset_out = out_dir / dataset_name
    ensure_dir(dataset_out)

    for p in tqdm(paths, desc="Extracting per-image feature + attention maps"):
        # load original for overlay
        img_pil = Image.open(p).convert("RGB")
        img_resized = transforms.Resize(args.img_size, interpolation=transforms.InterpolationMode.BICUBIC)(img_pil)
        img_cropped = transforms.CenterCrop(args.img_size)(img_resized)
        img_np = np.asarray(img_cropped).astype(np.float32) / 255.0  # (H,W,3)

        x = tfm(img_pil).unsqueeze(0).to(args.device)

        # Feature map: get patch tokens from last layer and reshape to (D,H',W')
        feats = model.forward_features(x)  # dict
        patch_tokens = feats["x_norm_patchtokens"]  # (B, num_patches, D)
        patch_tokens = patch_tokens[0].float().cpu().numpy()  # (P, D)

        # reshape patches to grid
        if patch_tokens.shape[0] != grid_h * grid_w:
            # If your img_size changes or model uses different tokenization, this guards it.
            raise RuntimeError(f"Unexpected num_patches={patch_tokens.shape[0]}, expected {grid_h*grid_w}")

        feat_map = patch_tokens.reshape(grid_h, grid_w, -1).transpose(2, 0, 1)  # (D, H', W')

        # Save feature map as .npy
        stem = p.stem
        (dataset_out / f"{stem}_feat.npy").write_bytes(b"")  # ensure path exists on Windows oddities
        np.save(dataset_out / f"{stem}_feat.npy", feat_map)

        # --- Attention map: recompute CLS->patch attention from cached qkv ---
        qkv = qkv_cache.get("qkv", None)
        if qkv is None:
            raise RuntimeError("qkv hook did not fire; attention cache is empty.")

        # qkv: (B, N, 3*C)
        B, N, threeC = qkv.shape
        C = threeC // 3
        num_heads = model.num_heads
        head_dim = C // num_heads
        scale = head_dim ** -0.5

        qkv = qkv.reshape(B, N, 3, num_heads, head_dim)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)  # (B, H, N, Dh)
        k = qkv[:, :, 1].permute(0, 2, 1, 3)  # (B, H, N, Dh)

        attn = (q @ k.transpose(-2, -1)) * scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)

        # token layout: [CLS, storage_tokens..., PATCHES...]
        patch_start = 1 + int(getattr(model, "n_storage_tokens", 0))
        cls_to_patches = attn[0, :, 0, patch_start:]  # (H, P)
        cls_to_patches = cls_to_patches.mean(axis=0)  # (P,)

        attn_map = cls_to_patches.reshape(grid_h, grid_w)
        attn_map = attn_map / (attn_map.max() + 1e-8)
        attn_map = attn_map.detach().cpu().numpy()

        # Upsample attention to image size for overlay
        attn_up = np.array(
            Image.fromarray(to_uint8(attn_map)).resize((args.img_size, args.img_size), resample=Image.BICUBIC)
        ).astype(np.float32) / 255.0

        # Save attention overlay
        plt.figure(figsize=(5, 5))
        plt.imshow(img_np)
        plt.imshow(attn_up, alpha=0.45)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(dataset_out / f"{stem}_attn.png", dpi=200)
        plt.close()

        # --- Optional: quick feature-map visualization via PCA -> 3 channels ---
        # Convert (D,H',W') to (H'*W', D), PCA->3, then upsample.
        fm = feat_map.transpose(1, 2, 0).reshape(-1, feat_map.shape[0])  # (P, D)
        fm3 = PCA(n_components=3, random_state=0).fit_transform(fm)  # (P,3)
        fm3 = (fm3 - fm3.min(axis=0)) / (np.ptp(fm3, axis=0) + 1e-8)
        fm3 = fm3.reshape(grid_h, grid_w, 3)
        fm3_up = np.array(
            Image.fromarray(to_uint8(fm3)).resize((args.img_size, args.img_size), resample=Image.BICUBIC)
        )

        Image.fromarray(fm3_up).save(dataset_out / f"{stem}_feat_rgb.png")

    hook_handle.remove()
    print(f"\nDone. Outputs saved under: {dataset_out}")


if __name__ == "__main__":
    main()
