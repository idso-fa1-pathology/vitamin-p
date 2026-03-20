"""
vitaminp/pretrained.py

Pretrained model registry and auto-download from HuggingFace Hub.

Usage:
    import vitaminp
    model = vitaminp.load_model('dual')   # VitaminPDual  - H&E nuclei + cell
    model = vitaminp.load_model('syn')    # VitaminPSyn   - Synthetic MIF
    model = vitaminp.load_model('flex')   # VitaminPFlex  - Flexible multi-channel
"""

import os
import torch
from pathlib import Path

# ── Registry ──────────────────────────────────────────────────────────────────
# Maps model name → (HuggingFace filename, model class name, model size)

HF_REPO = "idso-fa1-pathology/VitaminP"

MODEL_REGISTRY = {
    "dual": {
        "filename": "vitamin_p_dual.pth",
        "class":    "VitaminPDual",
        "size":     "base",
        "description": "H&E dual nuclei + cell segmentation (base)",
    },
    "syn": {
        "filename": "vitamin_p_dual.pth",   # syn and flex share the same weights
        "class":    "VitaminPSyn",
        "size":     "base",
        "description": "Synthetic MIF segmentation (base)",
    },
    "flex": {
        "filename": "vitamin_p_flex.pth",
        "class":    "VitaminPFlex",
        "size":     "large",
        "description": "Flexible multi-channel segmentation (large)",
    },
}

# ── Cache directory ────────────────────────────────────────────────────────────
def get_cache_dir() -> Path:
    """Returns ~/.cache/vitaminp, creating it if needed."""
    cache = Path.home() / ".cache" / "vitaminp"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


# ── Downloader ────────────────────────────────────────────────────────────────
def download_weights(filename: str) -> Path:
    """
    Downloads weights from HuggingFace Hub if not already cached.
    Returns the local path to the .pth file.
    """
    cache_dir = get_cache_dir()
    local_path = cache_dir / filename

    if local_path.exists():
        print(f"[vitaminp] Using cached weights: {local_path}")
        return local_path

    print(f"[vitaminp] Downloading weights '{filename}' from HuggingFace...")
    print(f"[vitaminp] Repo: {HF_REPO}")
    print(f"[vitaminp] This only happens once — weights are cached at {cache_dir}")

    try:
        # Try huggingface_hub first (cleanest)
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
            cache_dir=str(cache_dir),
            local_dir=str(cache_dir),
        )
        print(f"[vitaminp] ✅ Downloaded to {path}")
        return Path(path)

    except ImportError:
        # Fallback: plain requests download
        import requests
        url = f"https://huggingface.co/{HF_REPO}/resolve/main/{filename}"
        print(f"[vitaminp] (huggingface_hub not installed, using requests fallback)")
        print(f"[vitaminp] URL: {url}")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r[vitaminp] {pct:.1f}%", end="", flush=True)

        print(f"\n[vitaminp] ✅ Downloaded to {local_path}")
        return local_path


# ── Main entry point ──────────────────────────────────────────────────────────
def load_model(name: str, device: str = "cuda", map_location=None):
    """
    Load a pretrained VitaminP model by name.

    Args:
        name       : One of 'dual', 'syn', 'flex'
        device     : 'cuda' or 'cpu' (default: 'cuda', falls back to 'cpu')
        map_location: Optional torch map_location override

    Returns:
        model      : Pretrained model in eval() mode, moved to device

    Example:
        import vitaminp
        model = vitaminp.load_model('dual')
        model = vitaminp.load_model('flex', device='cpu')
    """
    name = name.lower().strip()

    if name not in MODEL_REGISTRY:
        valid = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"[vitaminp] Unknown model '{name}'. "
            f"Available models: {valid}\n"
            f"Example: vitaminp.load_model('dual')"
        )

    info = MODEL_REGISTRY[name]
    print(f"[vitaminp] Loading model: {name} — {info['description']}")

    # ── Resolve device ────────────────────────────────────────────────────────
    if map_location is None:
        if device == "cuda" and not torch.cuda.is_available():
            print("[vitaminp] ⚠️  CUDA not available, falling back to CPU.")
            device = "cpu"
        map_location = device

    # ── Import the right model class ──────────────────────────────────────────
    class_name = info["class"]
    size       = info["size"]

    try:
        from vitaminp.models import VitaminPDual, VitaminPFlex, VitaminPSyn
        model_classes = {
            "VitaminPDual": VitaminPDual,
            "VitaminPSyn":  VitaminPSyn,
            "VitaminPFlex": VitaminPFlex,
        }
        ModelClass = model_classes[class_name]
    except ImportError as e:
        raise ImportError(
            f"[vitaminp] Could not import model class '{class_name}'. "
            f"Make sure vitaminp is installed correctly.\nOriginal error: {e}"
        )

    # ── Download weights ──────────────────────────────────────────────────────
    weights_path = download_weights(info["filename"])

    # ── Build and load model ──────────────────────────────────────────────────
    model = ModelClass(model_size=size).to(map_location)
    state_dict = torch.load(weights_path, map_location=map_location)

    # Handle both raw state_dict and checkpoint dicts
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.eval()

    print(f"[vitaminp] ✅ Model '{name}' ready on {map_location}.")
    return model


# ── Helper: list available models ─────────────────────────────────────────────
def available_models():
    """Print all available pretrained models."""
    print("\n[vitaminp] Available pretrained models:")
    print(f"{'Name':<8} {'Class':<16} {'Size':<8} Description")
    print("-" * 60)
    for name, info in MODEL_REGISTRY.items():
        print(f"{name:<8} {info['class']:<16} {info['size']:<8} {info['description']}")
    print()
