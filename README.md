# VitaminP: Cell & Nuclei Segmentation for H&E and Multiplex IF

[![PyPI version](https://img.shields.io/pypi/v/vitaminp.svg)](https://pypi.org/project/vitaminp/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**VitaminP** is a deep learning framework for robust cell and nuclei segmentation in H&E and multiplex immunofluorescence (MIF) whole slide images (WSI). Built on DINOv2 U-Net, it supports automatic resolution matching, tissue detection, and joint H&E + MIF inference.

---

## 📦 Installation

```bash
pip install vitaminp
```

**Requirements:** Python 3.8+, PyTorch 2.0+, CUDA 11.8+ (for GPU)

> ⚠️ If you see a NumPy/OpenCV conflict, run: `pip install "numpy<2" --force-reinstall`

---

## 🚀 Quick Start

```python
import vitaminp

# See all available pretrained models
vitaminp.available_models()

# Load a pretrained model — downloads automatically on first use, cached forever after
model = vitaminp.load_model('dual')   # H&E nuclei + cell
model = vitaminp.load_model('flex')   # Flexible multi-channel (H&E or MIF)
model = vitaminp.load_model('syn')    # Synthetic MIF
```

| Model | Size | Best for |
|-------|------|----------|
| `dual` | base  | H&E whole slide images — joint nuclei + cell segmentation |
| `flex` | large | H&E or MIF — flexible multi-channel inputs |
| `syn`  | base  | Synthetic MIF generation and segmentation |

---

## 📖 Usage

### H&E — Joint Nuclei + Cell Segmentation (`dual`)

```python
import vitaminp
from vitaminp.inference import WSIPredictor

# 1. Load pretrained model
model = vitaminp.load_model('dual', device='cuda')

# 2. Initialize Predictor
predictor = WSIPredictor(
    model=model,
    device='cuda',
    patch_size=512,
    overlap=64,
    target_mpp=0.4250,
    magnification=20,
    batch_size=32,
    tissue_dilation=1,
)

# 3. Run Inference
# Passing both branches triggers Joint Inference mode —
# nuclei predictions constrain and improve cell segmentation
results = predictor.predict(
    wsi_path='slide.svs',
    output_dir='results/',
    branches=['he_nuclei', 'he_cell'],
    filter_tissue=True,
    tissue_threshold=0.10,
    clean_overlaps=True,
    save_geojson=True,
    save_parquet=False,
    simplify_epsilon=None,
    coord_precision=None,
    min_area_um=10.0,
)

print(f"✅ Nuclei: {results['he_nuclei']['num_detections']}")
print(f"✅ Cells:  {results['he_cell']['num_detections']}")
```

---

### Multiplex IF (MIF) — Flexible Multi-Channel (`flex`)

```python
import vitaminp
from vitaminp.inference import WSIPredictor
from vitaminp.inference.channel_config import ChannelConfig

# 1. Load pretrained model
model = vitaminp.load_model('flex', device='cuda')

# 2. Define channel mapping
config = ChannelConfig(
    nuclear_channel=2,                        # e.g. DAPI
    membrane_channel=[0, 1],                  # e.g. cell markers
    membrane_combination='max',               # combine via max projection
    channel_names={0: 'CellMarker1', 1: 'CellMarker2', 2: 'DAPI'}
)

# 3. Initialize Predictor
# 💡 batch_size: start with 16. For 24GB+ GPU try 32-64. 
#    If "CUDA Out of Memory", lower to 8 or 4.
predictor = WSIPredictor(
    model=model,
    device='cuda',
    patch_size=512,
    overlap=64,
    target_mpp=0.4250,
    magnification=20,
    mif_channel_config=config,
    batch_size=16,
)

# 4. Run Inference
results = predictor.predict(
    wsi_path='he_image.png',
    wsi_path_mif='mif_image.png',       # co-registered MIF
    output_dir='results/',
    branches=['he_nuclei', 'he_cell'],
    filter_tissue=True,
    clean_overlaps=True,
    save_geojson=True,
    save_visualization=True,
    detection_threshold=0.2,
    min_area_um=5.0,
)

print(f"✅ H&E nuclei: {results['he_nuclei']['num_detections']}")
print(f"✅ H&E cells:  {results['he_cell']['num_detections']}")
```

---

## 📊 Output Files

```
results/
├── he_nuclei_detections.geojson    # QuPath-compatible annotations
├── he_cell_detections.geojson
├── he_nuclei_boundaries.png        # Visualization with contours
└── he_nuclei_centroids.csv         # Centroid coordinates (optional)
```

GeoJSON output is directly compatible with [QuPath](https://qupath.github.io/).

---

## 🎯 Common Recipes

### Batch Processing
```python
import glob
from pathlib import Path
import vitaminp
from vitaminp.inference import WSIPredictor

model = vitaminp.load_model('dual', device='cuda')
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

### Override MPP (images without metadata)
```python
results = predictor.predict(
    wsi_path='image.png',
    mpp_override=0.25,
    branches=['he_nuclei'],
)
```

---

## 🔧 Troubleshooting

**CUDA Out of Memory**
```python
predictor = WSIPredictor(model=model, device='cuda', batch_size=4)  # lower batch_size
```

**No MPP in image metadata**
```python
results = predictor.predict(wsi_path='image.png', mpp_override=0.4250, ...)
```

**Too many false positives**
```python
results = predictor.predict(..., detection_threshold=0.7, min_area_um=10.0)
```

**NumPy / OpenCV conflict**
```bash
pip install "numpy<2" --force-reinstall
```

---

## 📚 Citation

If you use VitaminP in your research, please cite:
```bibtex
@article{vitaminp2025,
  title   = {VitaminP: Robust Cell Segmentation for H&E and Multiplex IF},
  author  = {Shokrollahi, Yousef},
  journal = {arXiv},
  year    = {2025}
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file.

## 🙋 Support

- 🐛 **Issues:** [GitHub Issues](https://github.com/yourusername/vitaminp/issues)
- 📧 **Email:** your.email@mdanderson.org

---

**Made with ❤️ for the computational pathology community**