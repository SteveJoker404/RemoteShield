import argparse
import os
from typing import Optional, Union
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def apply_cloud_fog(
    image: Union[str, Image.Image],
    strength: float,
    *,
    seed: Optional[int] = None,
    preserve_size: bool = True,
) -> Image.Image:
    """
    Add a natural-looking cloud/fog veil that reduces clarity.
    
    Args:
        image: File path (str) or PIL Image to process
        strength: 0.0 (no change) -> 1.0 (very strong fog/blur)
        seed: Random seed for reproducible results
        preserve_size: Preserve original image size (default: True)
    
    Returns:
        PIL.Image in RGB mode with cloud/fog effect applied
    
    Raises:
        ValueError: If strength is not in [0, 1]
        FileNotFoundError: If image path does not exist
        TypeError: If image is not a str or PIL.Image
    """
    # -------- validate inputs --------
    if not (0.0 <= float(strength) <= 1.0):
        raise ValueError(f"strength must be in [0, 1], got {strength}")

    # Load image
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image path does not exist: {image}")
        with Image.open(image) as src:
            img = src.convert("RGB")
    elif isinstance(image, Image.Image):
        img = image
    else:
        raise TypeError("image must be a file path (str) or a PIL.Image.Image")

    # Convert to RGB for consistent processing
    img = img.convert("RGB")
    w, h = img.size
    if w < 2 or h < 2:
        return img  # nothing meaningful to do

    if strength == 0.0:
        return img.copy()

    rng = np.random.default_rng(seed)

    # -------- helper: build low-frequency cloud mask --------
    def _cloud_mask(width: int, height: int, s: float) -> np.ndarray:
        """
        Build cloud-like mask using multi-octave noise.
        
        Returns:
            Float mask in [0,1] shaped (H,W) with cloud-like structures
        """
        # Choose a base resolution relative to image size and strength
        # Smaller base -> larger blobs. Increase blob size slightly with strength.
        base = int(max(16, min(width, height) / (8.0 + 10.0 * (1.0 - s))))
        base_w = max(8, int(width / base))
        base_h = max(8, int(height / base))

        mask = np.zeros((height, width), dtype=np.float32)

        # Multi-octave noise (3 layers): low, mid, higher frequency
        for octave, weight in [(1, 0.60), (2, 0.28), (4, 0.12)]:
            ow = max(4, base_w * octave)
            oh = max(4, base_h * octave)
            noise = rng.random((oh, ow), dtype=np.float32)

            # Upsample to image size (bilinear)
            noise_img = Image.fromarray((noise * 255).astype(np.uint8))
            noise_img = noise_img.resize((width, height), resample=Image.BILINEAR)

            # Blur to remove blockiness; blur radius scales with strength and octave
            blur_r = (1.5 + 6.0 * s) / octave
            noise_img = noise_img.filter(ImageFilter.GaussianBlur(radius=blur_r))

            noise_up = (np.asarray(noise_img).astype(np.float32) / 255.0)
            mask += weight * noise_up

        # Normalize to [0,1]
        mask -= mask.min()
        denom = mask.max() - mask.min()
        if denom > 1e-8:
            mask /= denom
        else:
            mask[:] = 0.0

        # Soft “cloudiness” shaping:
        # push mid-high values up a bit to form cloud patches but keep natural.
        # The shaping strength increases with s.
        gamma = 1.0 - 0.8 * s  # <1 brightens midtones
        mask = np.clip(mask, 0.0, 1.0) ** gamma

        # Optional gentle thresholding to avoid uniform haze everywhere at low strength
        # At small s, keep mask sparse; at large s, become more global.
        t = 0.55 - 0.25 * s
        mask = np.clip((mask - t) / (1.0 - t + 1e-8), 0.0, 1.0)

        return mask

    cloud = _cloud_mask(w, h, strength)  # (H,W) float in [0,1]

    # -------- apply fog veil --------
    img_np = np.asarray(img).astype(np.float32) / 255.0  # (H,W,3)

    # Fog color: slightly off-white (more natural than pure white)
    fog_color = np.array([0.90, 0.93, 0.98], dtype=np.float32)

    # Alpha controls how strong the veil is; keep it smooth and not too "painted"
    # At strength=1, alpha is strong but not instantly white-out before blur/contrast.
    alpha = 0.78 * strength  # up to ~0.78
    cloud3 = cloud[..., None]

    # Screen-like blend (brighter veil), but tempered to avoid fake look
    # First do a linear mix, then a mild screen to lift highlights naturally.
    base_mix = img_np * (1.0 - alpha * cloud3) + fog_color * (alpha * cloud3)
    screen = 1.0 - (1.0 - img_np) * (1.0 - (alpha * 0.55) * cloud3)
    out_np = 0.65 * base_mix + 0.35 * screen

    out_np = np.clip(out_np, 0.0, 1.0)

    out = Image.fromarray((out_np * 255.0).astype(np.uint8))

    # -------- reduce contrast slightly + lift brightness slightly (haze effect) --------
    # Keep this subtle at low strength, stronger at high.
    contrast_factor = 1.0 - 0.35 * strength  # down to 0.65
    brightness_factor = 1.0 + 0.08 * strength  # up to 1.08

    out = ImageEnhance.Contrast(out).enhance(contrast_factor)
    out = ImageEnhance.Brightness(out).enhance(brightness_factor)

    # -------- apply blur (clarity loss) --------
    # Blur radius scales with image size to be resolution-robust.
    # At strength=1, make it "very blurry" but not completely unreadable for typical sizes.
    base_r = min(w, h) / 220.0  # ~1.0 for 224px, ~4.0 for 900px
    blur_radius = (0.05 + 1.2 * strength) * base_r
    if blur_radius > 0.01:
        out = out.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Optional: preserve original size flag (kept for interface consistency)
    if preserve_size and out.size != (w, h):
        out = out.resize((w, h), resample=Image.BICUBIC)

    return out


# ------------------- Single-image CLI -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply the cloud/fog perturbation to a single image."
    )
    parser.add_argument(
        "--input-image",
        type=str,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--output-image",
        type=str,
        required=True,
        help="Path to save the perturbed image.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.45,
        help="Perturbation strength in [0, 1].",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible perturbations.",
    )
    parser.add_argument(
        "--no-preserve-size",
        action="store_true",
        help="Disable output-size preservation.",
    )
    args = parser.parse_args()

    fogged = apply_cloud_fog(
        args.input_image,
        strength=args.strength,
        seed=args.seed,
        preserve_size=not args.no_preserve_size,
    )
    output_dir = os.path.dirname(args.output_image)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fogged.save(args.output_image)
    print(f"Saved perturbed image to: {args.output_image}")
