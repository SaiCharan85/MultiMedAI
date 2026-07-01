"""Component 1: SYNTHESIS — Stable Diffusion v1.5, INFERENCE ONLY.

Embeddings used: SD v1.5's built-in CLIP ViT-L/14 text encoder. It turns the
prompt into token embeddings that condition the UNet's cross-attention. We use
it AS-IS (frozen) — no training of any kind.

WHY INFERENCE ONLY (no fine-tuning / no DreamBooth / no LoRA training):
  Diffusion training needs many GPU-hours. This machine is AMD Radeon + Windows,
  where PyTorch has NO usable GPU backend (ROCm is Linux-only, CUDA is NVIDIA-only),
  so every op runs on CPU. Training a UNet on CPU would take days-to-weeks, which
  violates the "minutes-to-an-hour per step" constraint. Inference is feasible
  (slow but functional) so we do inference only.

Subcommands:
  prepare : download SD v1.5 weights into weights/, and cache a small set of REAL
            pathology images (from open path-vqa) to act as the FID reference set.
  run     : generate demo images (one per configured prompt) + a FID generation set
            (histopathology prompt, varied seeds), all on CPU.
  eval    : compute REAL FID (pytorch-fid) between the real and generated sets.

Run:  python -m src.synthesis prepare | run | eval | all
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from src.common import load_config, get_device, set_seed, resolve, ensure_dir


def _paths(cfg):
    out = ensure_dir(resolve(cfg["paths"]["outputs"], "synthesis"))
    real_dir = ensure_dir(resolve(cfg["paths"]["data"], "fid_real"))
    gen_dir = ensure_dir(resolve(cfg["paths"]["outputs"], "synthesis", "fid_gen"))
    weights = ensure_dir(resolve(cfg["paths"]["weights"], "sd15"))
    return out, real_dir, gen_dir, weights


# ---------------------------------------------------------------------------
def prepare(cfg):
    """Download SD v1.5 + cache real pathology reference images for FID."""
    from diffusers import StableDiffusionPipeline
    import torch

    _, real_dir, _, weights = _paths(cfg)
    scfg = cfg["synthesis"]

    print(f"[prepare] Downloading {scfg['model_id']} (~4GB, one time)...")
    # Download + cache into weights/sd15 so nothing lands in git.
    StableDiffusionPipeline.from_pretrained(
        scfg["model_id"],
        torch_dtype=torch.float32,   # CPU => float32 (no fp16 on CPU)
        safety_checker=None,
        cache_dir=str(weights),
    )
    print(f"[prepare] SD v1.5 cached under {weights}")

    # Cache REAL pathology images as the FID reference set (open data, no creds).
    n = scfg["fid"]["real_subset"]
    existing = list(real_dir.glob("*.png"))
    if len(existing) >= n:
        print(f"[prepare] {len(existing)} real FID images already cached.")
        return
    print(f"[prepare] Caching {n} REAL pathology images from path-vqa for FID...")
    from datasets import load_dataset

    ds = load_dataset(cfg["vqa"]["dataset"], split="train", streaming=True)
    size = scfg["image_size"]
    saved = 0
    for ex in ds:
        img = ex["image"].convert("RGB").resize((size, size))
        img.save(real_dir / f"real_{saved:03d}.png")
        saved += 1
        if saved >= n:
            break
    print(f"[prepare] Saved {saved} real reference images to {real_dir}")


# ---------------------------------------------------------------------------
def _load_pipe(cfg):
    from diffusers import StableDiffusionPipeline
    import torch

    scfg = cfg["synthesis"]
    _, _, _, weights = _paths(cfg)
    device = get_device()
    print(f"[run] Loading SD v1.5 on device={device} (float32)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        scfg["model_id"],
        torch_dtype=torch.float32,
        safety_checker=None,
        cache_dir=str(weights),
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def run(cfg):
    """Generate demo images + FID generation set on CPU. Times each image."""
    import torch

    scfg = cfg["synthesis"]
    out_dir, _, gen_dir, _ = _paths(cfg)
    device = get_device()
    pipe = _load_pipe(cfg)

    steps = scfg["num_inference_steps"]
    size = scfg["image_size"]
    gscale = scfg["guidance_scale"]

    def gen_one(prompt, seed):
        g = torch.Generator(device=device).manual_seed(seed)
        t0 = time.time()
        img = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=gscale,
            height=size,
            width=size,
            generator=g,
        ).images[0]
        return img, time.time() - t0

    # --- demo gallery: one image per configured prompt ---
    print(f"\n[run] Generating {len(scfg['prompts'])} demo images "
          f"({size}px, {steps} steps) on CPU...")
    times = []
    for i, prompt in enumerate(scfg["prompts"]):
        img, dt = gen_one(prompt, cfg["seed"] + i)
        img.save(out_dir / f"demo_{i:02d}.png")
        times.append(dt)
        print(f"  demo_{i:02d}  {dt:6.1f}s  | {prompt[:48]}")

    # --- FID generation set: same histopath prompt, varied seeds ---
    if scfg["fid"]["enabled"]:
        n = scfg["fid"]["gen_subset"]
        fid_prompt = scfg["fid"]["fid_prompt"]
        print(f"\n[run] Generating {n} FID images from histopath prompt...")
        for j in range(n):
            img, dt = gen_one(fid_prompt, 1000 + j)
            img.save(gen_dir / f"gen_{j:03d}.png")
            times.append(dt)
            print(f"  gen_{j:03d}   {dt:6.1f}s")

    print(f"\n[run] Done. {len(times)} images, "
          f"avg {sum(times)/len(times):.1f}s/img on CPU.")
    print(f"[run] Demo images -> {out_dir}")
    print(f"[run] FID gen set -> {gen_dir}")


# ---------------------------------------------------------------------------
def evaluate(cfg):
    """Compute REAL FID between real and generated sets via pytorch-fid."""
    _, real_dir, gen_dir, _ = _paths(cfg)
    device = get_device()

    real_imgs = list(real_dir.glob("*.png"))
    gen_imgs = list(gen_dir.glob("*.png"))
    print(f"[eval] FID inputs: {len(real_imgs)} real vs {len(gen_imgs)} generated")
    if len(real_imgs) < 2 or len(gen_imgs) < 2:
        print("[eval] Not enough images for FID. Run prepare + run first.")
        return None

    from pytorch_fid.fid_score import calculate_fid_given_paths

    # dims=2048 is the standard Inception pool layer.
    fid = calculate_fid_given_paths(
        [str(real_dir), str(gen_dir)],
        batch_size=8,
        device=device,
        dims=2048,
    )
    print("=" * 60)
    print(f"  REAL FID = {fid:.3f}")
    print(f"  (computed by pytorch-fid on {len(real_imgs)} real / "
          f"{len(gen_imgs)} generated images, CPU)")
    print("  NOTE: small-sample FID is HIGH-VARIANCE and biased upward; this")
    print("  is a consumer-CPU sanity figure, NOT a full-scale benchmark.")
    print("=" * 60)

    # persist the real number so the README/app can show it honestly
    metrics_file = resolve(cfg["paths"]["outputs"], "synthesis", "fid_metric.txt")
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write(f"FID={fid:.3f}\n")
        f.write(f"real_images={len(real_imgs)}\n")
        f.write(f"generated_images={len(gen_imgs)}\n")
        f.write("device=cpu\n")
    print(f"[eval] Wrote {metrics_file}")
    return fid


# ---------------------------------------------------------------------------
def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("prepare", "all"):
        prepare(cfg)
    if cmd in ("run", "all"):
        run(cfg)
    if cmd in ("eval", "all"):
        evaluate(cfg)
    if cmd not in ("prepare", "run", "eval", "all"):
        print("usage: python -m src.synthesis [prepare|run|eval|all]")


if __name__ == "__main__":
    main()
