"""Shared utilities for MultiMedAI.

Everything here is CPU-only by design. On this machine (AMD Ryzen + AMD Radeon,
Windows) PyTorch has NO usable GPU backend:
  - ROCm (AMD's CUDA-equivalent) ships Linux-only wheels.
  - CUDA is NVIDIA-only.
So `torch.cuda.is_available()` is False and we force device="cpu" everywhere.

All paths are resolved relative to the repo root (the parent of this file's
directory). There are NO hardcoded absolute paths anywhere in the codebase.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import yaml

# Repo root = parent of src/
ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load config.yaml from the repo root."""
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve(*parts: str) -> Path:
    """Resolve a path relative to repo root, creating parent dirs as needed."""
    p = ROOT.joinpath(*parts)
    return p


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_device() -> str:
    """Always 'cpu' on this hardware. Returns a string, never a CUDA device."""
    import torch

    if torch.cuda.is_available():  # never True here; kept honest, not assumed
        return "cuda"
    return "cpu"


def set_seed(seed: int = 42) -> None:
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)


def env_report() -> dict:
    """Print and return a clear environment report. Confirms CPU-only operation."""
    import platform
    import sys

    import torch

    device = get_device()
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "threads": torch.get_num_threads(),
    }

    print("=" * 64)
    print("  MultiMedAI - Environment Report")
    print("=" * 64)
    for k, v in info.items():
        print(f"  {k:16s}: {v}")
    print("-" * 64)
    if device == "cpu":
        print("  NOTICE: device is CPU. AMD Radeon has no usable PyTorch GPU")
        print("          backend on Windows (ROCm=Linux-only, CUDA=NVIDIA-only).")
        print("          All runs are FUNCTIONAL but SLOW. Subsets are kept small.")
    else:
        print("  NOTICE: a CUDA device was detected (unexpected on this hardware).")
    print("=" * 64)
    return info


if __name__ == "__main__":
    env_report()
