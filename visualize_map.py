import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

import run_config as cfg
from src.datasets import PascalVOC
from src.model import BFOR_model
from src.train import load_model


def preprocess_image(image_input):
    """
    Preprocesses an image (file path, PIL Image, or torch Tensor) to match the model input:
    - Symmetric padding to square canvas (black background)
    - Resize to 448x448
    - Convert to normalized PyTorch tensor [1, 3, 448, 448] in [0, 1]

    Returns:
        tensor_img: [1, 3, 448, 448] torch.Tensor
        display_img: [448, 448, 3] numpy array in [0, 1] for visualization
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    elif isinstance(image_input, torch.Tensor):
        if image_input.dim() == 3:
            tensor = image_input.unsqueeze(0)
        else:
            tensor = image_input
        display = tensor[0].permute(1, 2, 0).cpu().numpy()
        return tensor, display
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    w, h = img.size
    s = max(w, h)

    pad_left = (s - w) // 2
    pad_right = s - w - pad_left
    pad_top = (s - h) // 2
    pad_bottom = s - h - pad_top

    padded_img = F.pad(
        img, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=0
    )
    resized_img = F.resize(padded_img, [448, 448])
    tensor_img = F.to_tensor(resized_img).unsqueeze(0)
    display_img = np.array(resized_img, dtype=np.float32) / 255.0

    return tensor_img, display_img


def extract_all_maps(model, image_input, device=None, pixel_scale=True):
    """
    Feeds an image to the model and extracts all 9 output maps:
    For each scale ('sml', 'med', 'lrg'):
        - 'obj': Objectness / ROS heatmap [448, 448] in [0, 1]
        - 'w': Width map [448, 448] (in pixels if pixel_scale else [0, 1])
        - 'h': Height map [448, 448] (in pixels if pixel_scale else [0, 1])

    Returns:
        maps: dict with structure maps[scale][head] -> 2D numpy array [448, 448]
        display_img: [448, 448, 3] numpy array
    """
    if device is None:
        device = cfg.DEVICE if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    tensor_img, display_img = preprocess_image(image_input)
    tensor_img = tensor_img.to(device)

    with torch.no_grad():
        with torch.amp.autocast(
            device_type=device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        ):
            out = model(tensor_img)

    scale_mult = 448.0 if pixel_scale else 1.0

    maps = {}
    for scale in ["sml", "med", "lrg"]:
        maps[scale] = {
            "obj": out[scale]["obj"][0, 0].cpu().to(torch.float32).numpy(),
            "w": (out[scale]["w"][0, 0].cpu().to(torch.float32).numpy()) * scale_mult,
            "h": (out[scale]["h"][0, 0].cpu().to(torch.float32).numpy()) * scale_mult,
        }

    return maps, display_img


def visualize_9_maps(
    maps,
    save_path="heatmaps.png",
    colormap="jet",
    pixel_scale=True,
    show=False,
):
    """
    Plots a 3x3 grid containing exactly the 9 output heatmaps:
    - Row 1: Small scale (P3)  -> Objectness, Width, Height
    - Row 2: Medium scale (P4) -> Objectness, Width, Height
    - Row 3: Large scale (P5)  -> Objectness, Width, Height
    """
    scales = [("sml", "Small Scale (P3)"), ("med", "Medium Scale (P4)"), ("lrg", "Large Scale (P5)")]
    heads = [("obj", "Objectness"), ("w", "Width"), ("h", "Height")]

    unit_label = "px" if pixel_scale else "norm"

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    for row_idx, (scale_key, scale_name) in enumerate(scales):
        for col_idx, (head_key, head_name) in enumerate(heads):
            ax = axes[row_idx, col_idx]
            data = maps[scale_key][head_key]

            if head_key == "obj":
                # Objectness is bounded in [0, 1]
                im = ax.imshow(data, cmap=colormap, vmin=0.0, vmax=1.0)
                title = f"{scale_name} - {head_name}\n[min: {data.min():.3f}, max: {data.max():.3f}]"
            else:
                # Scale heads: show with their actual range
                vmax = 448.0 if pixel_scale else 1.0
                im = ax.imshow(data, cmap=colormap, vmin=0.0, vmax=vmax)
                title = f"{scale_name} - {head_name} ({unit_label})\n[min: {data.min():.1f}, max: {data.max():.1f}]"

            ax.set_title(title, fontsize=12, weight="bold")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("B-FOR Multi-Scale Outputs (3 Scales x 3 Heads = 9 Heatmaps)", fontsize=16, weight="bold", y=0.99)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"9-heatmap plot successfully saved to: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the 9 output heatmaps (3 heads x 3 scales) of B-FOR on an image."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an image file. If omitted, a sample from Pascal VOC test set is used.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Index in the Pascal VOC test set (used if --image is not provided).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.pt",
        help="Path to trained model weights (default: best_model.pt).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="heatmaps.png",
        help="Output file path for saving the figure (default: heatmaps.png).",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="jet",
        help="Matplotlib colormap (e.g. jet, inferno, viridis, magma).",
    )
    parser.add_argument(
        "--normalized-scale",
        action="store_true",
        help="Display width and height normalized in [0, 1] instead of pixels in [0, 448].",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively using plt.show().",
    )

    args = parser.parse_args()

    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"

    # 1. Initialize model and load weights
    model = BFOR_model(n_channels=cfg.N_CHANNELS, drop_rate=cfg.DROP_RATE).to(device)
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        load_model(model, path=args.checkpoint)
    else:
        print(f"[Warning] Checkpoint '{args.checkpoint}' not found. Using uninitialized weights.")

    # 2. Select image input
    if args.image:
        image_input = args.image
        print(f"Using image from file: {image_input}")
    else:
        print("Loading Pascal VOC test set...")
        test_set = PascalVOC(path=cfg.DATA_PATH, split="test")

        if args.index is not None:
            idx = args.index
        else:
            # Pick a sample that contains unseen objects if possible
            idx = 0
            for i in range(min(500, len(test_set))):
                _, target, _ = test_set[i]
                if len(target) > 0:
                    idx = i
                    break

        print(f"Using Pascal VOC test set sample at index {idx}")
        image_input, target, labels = test_set[idx]
        print(f"Ground truth labels in image: {labels}")

    # 3. Extract all 9 maps
    pixel_scale = not args.normalized_scale
    maps, _ = extract_all_maps(
        model=model,
        image_input=image_input,
        device=device,
        pixel_scale=pixel_scale,
    )

    # 4. Print stats to console
    unit = "px" if pixel_scale else "norm"
    print("\nSummary of all 9 Output Heatmaps:")
    for scale in ["sml", "med", "lrg"]:
        print(f"--- Scale: {scale.upper()} ---")
        for head in ["obj", "w", "h"]:
            m = maps[scale][head]
            print(f"  {head.upper():3s}: min = {m.min():.4f}, max = {m.max():.4f}, mean = {m.mean():.4f} {unit if head != 'obj' else ''}")

    # 5. Render the 3x3 grid
    visualize_9_maps(
        maps=maps,
        save_path=args.save_path,
        colormap=args.colormap,
        pixel_scale=pixel_scale,
        show=args.show,
    )


if __name__ == "__main__":
    main()
