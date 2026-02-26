# VitaminP: Cell & Nuclei Segmentation for H&E and Multiplex IF

[![PyPI version](https://img.shields.io/pypi/v/vitaminp.svg)](https://pypi.org/project/vitaminp/)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-blue.svg)](https://ghcr.io/idso-fa1-pathology/vitaminp)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**VitaminP** is a cross-modal deep learning framework for cell and nuclei segmentation in H&E and multiplex immunofluorescence (MIF) whole slide images. Built on DINOv2 vision transformers, it learns from paired H&E–MIF data to infer cytoplasmic boundaries that are invisible in standard brightfield microscopy — enabling whole-cell segmentation directly from H&E.

Trained on **14 public datasets across 34 cancer types and 7M+ annotated instances**.

---

## 📦 Installation

**Option 1 — pip (Python API):**
```bash
pip install vitaminp
```
> ⚠️ If you see a NumPy/OpenCV conflict: `pip install "numpy<2" --force-reinstall`

**Option 2 — Docker (recommended for HPC/servers):**
```bash
docker pull ghcr.io/idso-fa1-pathology/vitaminp:latest
```
> Includes all dependencies, pretrained weights, and CUDA 12.1 support out of the box.

---

## 🗺️ Which Model Should I Use?

| Model | Input | Best For | Speed |
|-------|-------|----------|-------|
| `flex` | H&E **or** MIF (any channel) | General purpose — most users start here | ⚡⚡⚡ Fastest |
| `dual` | H&E **+** MIF (paired) | Best whole-cell accuracy when both modalities available | ⚡⚡ |
| `syn`  | H&E only | H&E whole-cell when no MIF available | ⚡⚡ |

**What branch should I run?**

| Goal | `branches=` |
|------|-------------|
| H&E nuclei only | `['he_nuclei']` |
| H&E cells only | `['he_cell']` |
| H&E both (recommended) | `['he_nuclei', 'he_cell']` — nuclei constrain cells for better accuracy |
| MIF nuclei only | `['mif_nuclei']` |
| MIF cells only | `['mif_cell']` |
| MIF both (recommended) | `['mif_nuclei', 'mif_cell']` |
| All branches | `['he_nuclei', 'he_cell', 'mif_nuclei', 'mif_cell']` |

---

## 🚀 Quick Start

```python
import vitaminp

model = vitaminp.load_model('flex')   # downloads once, cached forever
vitaminp.available_models()           # list all models
```

---

## 📖 Python API Usage

### 1. Flex — General Purpose (H&E or MIF)

**H&E input** (most common):

```python
import vitaminp
from vitaminp.inference import WSIPredictor

model = vitaminp.load_model('flex', device='cuda')

predictor = WSIPredictor(
    model=model,
    device='cuda',
    patch_size=512,
    overlap=64,
    target_mpp=0.4250,
    magnification=20,
    batch_size=32,        # lower to 4-8 if out of memory
    tissue_dilation=1,
)

results = predictor.predict(
    wsi_path='slide.svs',
    output_dir='results/',
    branches=['he_nuclei', 'he_cell'],  # or ['he_nuclei'], ['he_cell']
    filter_tissue=True,
    tissue_threshold=0.10,
    clean_overlaps=True,
    save_geojson=True,
    min_area_um=10.0,
)

print(f"✅ Nuclei: {results['he_nuclei']['num_detections']}")
print(f"✅ Cells:  {results['he_cell']['num_detections']}")
```

**MIF input** — set channel config so the model knows which channels are nucleus vs membrane:

```python
import vitaminp
from vitaminp.inference import WSIPredictor
from vitaminp.inference.channel_config import ChannelConfig

model = vitaminp.load_model('flex', device='cuda')

config = ChannelConfig(
    nuclear_channel=2,                  # e.g. DAPI
    membrane_channel=[0, 1],            # e.g. cell markers
    membrane_combination='max',
    channel_names={0: 'CellMarker1', 1: 'CellMarker2', 2: 'DAPI'}
)

predictor = WSIPredictor(
    model=model,
    device='cuda',
    patch_size=512,
    overlap=64,
    target_mpp=0.4250,
    magnification=20,
    mif_channel_config=config,          # required for MIF input
    batch_size=16,
)

results = predictor.predict(
    wsi_path='mif_image.tif',
    output_dir='results/',
    branches=['mif_nuclei', 'mif_cell'],   # use mif_* branches for MIF input
    filter_tissue=True,
    clean_overlaps=True,
    save_geojson=True,
    save_visualization=True,
    detection_threshold=0.2,
    min_area_um=5.0,
)

print(f"✅ MIF Nuclei: {results['mif_nuclei']['num_detections']}")
print(f"✅ MIF Cells:  {results['mif_cell']['num_detections']}")
```

---

### 2. Dual — Paired H&E + MIF (best whole-cell accuracy)

Use this when you have **co-registered H&E and MIF** from the same tissue section. The model fuses both signals to resolve cytoplasmic boundaries that are ambiguous in H&E alone.

```python
import vitaminp
from vitaminp.inference import WSIPredictor
from vitaminp.inference.channel_config import ChannelConfig

model = vitaminp.load_model('dual', device='cuda')

config = ChannelConfig(
    nuclear_channel=2,
    membrane_channel=[0, 1],
    membrane_combination='max',
    channel_names={0: 'CellMarker1', 1: 'CellMarker2', 2: 'DAPI'}
)

predictor = WSIPredictor(
    model=model,
    device='cuda',
    patch_size=512,
    overlap=64,
    target_mpp=0.4250,
    magnification=20,
    mif_channel_config=config,
    batch_size=4,
)

results = predictor.predict(
    wsi_path='he_image.png',              # H&E
    wsi_path_mif='mif_image.png',         # co-registered MIF
    output_dir='results/',
    branches=['he_nuclei', 'he_cell', 'mif_nuclei', 'mif_cell'],
    filter_tissue=True,
    clean_overlaps=True,
    save_geojson=True,
    save_visualization=True,
    detection_threshold=0.2,
    min_area_um=5.0,
)

print(f"✅ H&E nuclei:  {results['he_nuclei']['num_detections']}")
print(f"✅ H&E cells:   {results['he_cell']['num_detections']}")
print(f"✅ MIF nuclei:  {results['mif_nuclei']['num_detections']}")
print(f"✅ MIF cells:   {results['mif_cell']['num_detections']}")
```

---

## 🐳 Docker Usage

The Docker image has everything pre-installed: CUDA 12.1, all dependencies, and pretrained weights baked in at `/workspace/checkpoints/`.

### Pull the image
```bash
docker pull ghcr.io/idso-fa1-pathology/vitaminp:latest
```

### Single image inference
```bash
docker run --gpus all --rm \
  -v /your/images:/data \
  -v /your/results:/results \
  ghcr.io/idso-fa1-pathology/vitaminp:latest \
  python3 /workspace/scripts/run_wsi_inference.py \
    --model_type flex \
    --model_size large \
    --checkpoint /workspace/checkpoints/vitamin_p_flex.pth \
    --wsi_path /data/slide.svs \
    --output_dir /results \
    --branches he_nuclei he_cell \
    --target_mpp 0.4250 \
    --magnification 20 \
    --batch_size 32 \
    --filter_tissue \
    --tissue_threshold 0.10 \
    --clean_overlaps \
    --save_geojson \
    --min_area_um 10.0
```

### Batch folder inference
```bash
docker run --gpus all --rm \
  -v /your/images:/data \
  -v /your/results:/results \
  ghcr.io/idso-fa1-pathology/vitaminp:latest \
  python3 /workspace/scripts/run_wsi_inference.py \
    --model_type flex \
    --model_size large \
    --checkpoint /workspace/checkpoints/vitamin_p_flex.pth \
    --wsi_folder /data \
    --wsi_extension svs \
    --output_dir /results \
    --branches he_nuclei he_cell \
    --target_mpp 0.4250 \
    --magnification 20 \
    --batch_size 32 \
    --filter_tissue \
    --save_geojson \
    --min_area_um 10.0
```

### MIF inference with Docker
```bash
docker run --gpus all --rm \
  -v /your/images:/data \
  -v /your/results:/results \
  ghcr.io/idso-fa1-pathology/vitaminp:latest \
  python3 /workspace/scripts/run_wsi_inference.py \
    --model_type flex \
    --model_size large \
    --checkpoint /workspace/checkpoints/vitamin_p_flex.pth \
    --wsi_path /data/mif_image.tif \
    --output_dir /results \
    --branches mif_nuclei mif_cell \
    --mif_nuclear_channel 2 \
    --mif_membrane_channels 0,1 \
    --mif_membrane_combination max \
    --target_mpp 0.4250 \
    --magnification 20 \
    --batch_size 16 \
    --filter_tissue \
    --save_geojson \
    --min_area_um 5.0
```

### Dual model (H&E + MIF) with Docker
```bash
docker run --gpus all --rm \
  -v /your/images:/data \
  -v /your/results:/results \
  ghcr.io/idso-fa1-pathology/vitaminp:latest \
  python3 /workspace/scripts/run_wsi_inference.py \
    --model_type dual \
    --model_size base \
    --checkpoint /workspace/checkpoints/vitamin_p_dual.pth \
    --wsi_path /data/he_image.png \
    --output_dir /results \
    --branches he_nuclei he_cell mif_nuclei mif_cell \
    --mif_nuclear_channel 2 \
    --mif_membrane_channels 0,1 \
    --target_mpp 0.4250 \
    --magnification 20 \
    --batch_size 4 \
    --filter_tissue \
    --save_geojson \
    --min_area_um 5.0
```

### Key Docker CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_type` | required | `flex` or `dual` |
| `--model_size` | required | `base` or `large` |
| `--checkpoint` | required | Path to `.pth` file |
| `--wsi_path` | — | Single image |
| `--wsi_folder` | — | Folder of images |
| `--branches` | `he_nuclei` | Space-separated: `he_nuclei he_cell mif_nuclei mif_cell` |
| `--target_mpp` | `0.25` | Microns per pixel |
| `--magnification` | `40` | `20` or `40` |
| `--batch_size` | auto | Lower if out of memory |
| `--mif_nuclear_channel` | — | Required for MIF input |
| `--mif_membrane_channels` | — | Comma-separated, e.g. `0,1` |
| `--detection_threshold` | `0.5` | Higher = fewer false positives |
| `--min_area_um` | `5.0` | Filter small detections (μm²) |
| `--save_geojson` | True | QuPath-compatible output |
| `--save_parquet` | False | Fast binary format |
| `--save_visualization` | True | PNG overlay images |

---

## 📊 Output Files

```
results/
├── he_nuclei_detections.geojson    # QuPath-compatible annotations
├── he_cell_detections.geojson
├── mif_nuclei_detections.geojson   # (if MIF branches used)
├── mif_cell_detections.geojson
├── he_nuclei_boundaries.png        # Visualization overlay
└── inference.log                   # Full run log
```

GeoJSON output is directly compatible with [QuPath](https://qupath.github.io/).

---

## 🎯 Common Recipes

### Batch Processing (Python)
```python
import glob
from pathlib import Path
import vitaminp
from vitaminp.inference import WSIPredictor

model = vitaminp.load_model('flex', device='cuda')
predictor = WSIPredictor(model=model, device='cuda', batch_size=32)

for slide_path in glob.glob('slides/*.svs'):
    name = Path(slide_path).stem
    results = predictor.predict(
        wsi_path=slide_path,
        output_dir=f'results/{name}',
        branches=['he_nuclei', 'he_cell'],
        save_geojson=True,
        min_area_um=10.0,
    )
    print(f"{name}: {results['he_nuclei']['num_detections']} nuclei")
```

### Image Without MPP Metadata
```python
results = predictor.predict(
    wsi_path='image.png',
    mpp_override=0.4250,
    branches=['he_nuclei'],
)
```

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| CUDA out of memory | Lower `batch_size` to 4–8 |
| No MPP in metadata | Add `mpp_override=0.4250` (Python) or `--wsi_properties '{"slide_mpp": 0.4250}'` (Docker) |
| Too many false positives | Increase `detection_threshold=0.7`, `min_area_um=10.0` |
| NumPy/OpenCV error | `pip install "numpy<2" --force-reinstall` |
| MIF channels wrong | Set `mif_channel_config` (Python) or `--mif_nuclear_channel` (Docker) |
| Docker: stale file handle | Use `--output_dir /tmp/results` then copy out after |
| Docker: no internet for backbone | Mount cache: `-v ~/.cache/huggingface:/root/.cache/huggingface` |

---

## 📚 Citation

If you use VitaminP in your research, please cite:

```bibtex
@article{shokrollahi2025vitaminp,
  title   = {Vitamin-P: vision transformer assisted multi-modality integration 
             network for pathology cell segmentation},
  author  = {Shokrollahi, Yasin and Pinao Gonzales, Karina and Barrientos Toro, Elizve 
             and Acosta, Paul and Chen, Pingjun and Yuan, Yinyin and Pan, Xiaoxi},
  journal = {arXiv},
  year    = {2025}
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file.

## 🙋 Support

- 🐛 **Issues:** [GitHub Issues](https://github.com/idso-fa1-pathology/vitamin-p/issues)
- 📧 **Contact:** MD Anderson Cancer Center — Department of Translational Molecular Pathology

---

**Made with ❤️ at MD Anderson Cancer Center**
