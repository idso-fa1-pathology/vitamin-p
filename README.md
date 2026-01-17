# VitaminP: Cell & Nuclei Segmentation for H&E and Multiplex IF

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**VitaminP** is a deep learning model for robust cell and nuclei segmentation in H&E and multiplex immunofluorescence (MIF) images. Supports whole slide images (WSI) with automatic resolution matching and tissue detection.

---

## 🚀 Quick Start (30 seconds)
```python
import torch
from vitaminp import VitaminPFlex
from vitaminp.inference import WSIPredictor

# Load model
model = VitaminPFlex(model_size='large').to('cuda')
model.load_state_dict(torch.load("checkpoints/vitamin_p_flex_large.pth"))
model.eval()

# Run inference
predictor = WSIPredictor(model=model, device='cuda')
results = predictor.predict(
    wsi_path='slide.svs',
    output_dir='results',
    branch='he_nuclei',
    save_geojson=True
)

print(f"✅ Found {results['num_detections']} nuclei in {results['processing_time']:.2f}s")
```

**That's it!** Results saved to `results/` with GeoJSON annotations and visualizations.

---

## 📦 Installation
```bash
# Clone repository
git clone https://github.com/yourusername/vitaminp.git
cd vitaminp

# Install dependencies
pip install -e .
```

**Requirements:** Python 3.8+, PyTorch 2.0+, CUDA 11.8+ (for GPU)

---

## 📖 Basic Usage

### **H&E Nuclei Detection**
```python
import torch
from vitaminp import VitaminPFlex
from vitaminp.inference import WSIPredictor

# Setup model
device = 'cuda'
model = VitaminPFlex(model_size='large').to(device)
model.load_state_dict(torch.load("checkpoints/vitamin_p_flex_large_fold2_best.pth"))
model.eval()

# Create predictor
predictor = WSIPredictor(
    model=model,
    device='cuda',
    patch_size=512,
    overlap=64,
    target_mpp=0.25,      # Auto-detected from file if available
    magnification=40
)

# Run inference
results = predictor.predict(
    wsi_path='slide.svs',
    output_dir='results',
    branch='he_nuclei',
    filter_tissue=True,           # Skip background tiles
    tissue_threshold=0.1,         # 10% minimum tissue
    clean_overlaps=True,          # Remove duplicates at tile boundaries
    save_geojson=True,            # Save annotations
    save_visualization=True,      # Save overlay images
    detection_threshold=0.5,      # Binary threshold (0.5-0.8)
    min_area_um=3.0,             # Filter small artifacts (μm²)
)

print(f"✅ Found {results['num_detections']} nuclei")
print(f"   Output: {results['output_dir']}")
```

---

### **Multiplex IF (MIF) Segmentation**
```python
from vitaminp.inference import ChannelConfig

# Define channel mapping
config = ChannelConfig(
    nuclear_channel=0,           # DAPI/SYTO channel
    membrane_channel=[1, 2],     # Membrane markers
    membrane_combination='max',  # Combine channels via max projection
    channel_names={0: 'SYTO13', 1: 'Cy3', 2: 'TexasRed'}
)

# Create predictor with MIF config
predictor = WSIPredictor(
    model=model,
    device='cuda',
    mif_channel_config=config,
    target_mpp=0.5,
    magnification=20
)

# Run MIF inference
results = predictor.predict(
    wsi_path='mif_image.tif',
    output_dir='results_mif',
    branch='he_nuclei',          # Uses same model weights
    save_geojson=True,
    min_area_um=5.0
)
```

---

### **Dual Modality (H&E + MIF)**

Use MIF predictions (cleaner) with H&E visualization:
```python
from vitaminp import VitaminPDual

# Load dual model
model = VitaminPDual(model_size='base').to('cuda')
model.load_state_dict(torch.load("checkpoints/vitamin_p_dual_base.pth"))
model.eval()

# Setup predictor
predictor = WSIPredictor(
    model=model,
    device='cuda',
    mif_channel_config=config
)

# Process both modalities
results = predictor.predict(
    wsi_path='he_image.png',           # H&E image
    wsi_path_mif='mif_image.png',      # Co-registered MIF
    output_dir='results_dual',
    branches=['he_nuclei', 'he_cell', 'mif_nuclei', 'mif_cell'],
    save_geojson=True
)

# H&E results now use high-quality MIF predictions automatically!
print(f"H&E nuclei: {results['he_nuclei']['num_detections']}")
print(f"MIF nuclei: {results['mif_nuclei']['num_detections']}")
```

**Key feature:** When using dual models, H&E branches automatically use MIF predictions (better quality) while keeping H&E background for visualization.

---

## 📊 Output Files

Running inference creates the following files:
```
results/
├── nuclei_detections.geojson    # QuPath-compatible annotations
├── nuclei_detections.json       # Raw instance data
├── nuclei_boundaries.png        # Visualization with contours
└── nuclei_centroids.csv         # (optional) Centroid coordinates
```

**GeoJSON format** is compatible with [QuPath](https://qupath.github.io/) for interactive viewing.

---

## 🎯 Common Recipes

### **Process Multiple Branches**
```python
results = predictor.predict(
    wsi_path='slide.svs',
    branches=['he_nuclei', 'he_cell'],  # Process both
    output_dir='results'
)

print(f"Nuclei: {results['he_nuclei']['num_detections']}")
print(f"Cells: {results['he_cell']['num_detections']}")
```

---

### **Override MPP (for images without metadata)**
```python
results = predictor.predict(
    wsi_path='image.png',
    mpp_override=0.25,  # Force 0.25 μm/pixel
    branch='he_nuclei'
)
```

---

### **Custom Area Filtering**
```python
results = predictor.predict(
    wsi_path='slide.svs',
    branch='he_nuclei',
    min_area_um=5.0,           # Filter nuclei < 5 μm²
    detection_threshold=0.6     # Higher threshold = fewer false positives
)
```

---

### **Batch Processing**
```python
import glob
from pathlib import Path

slides = glob.glob('slides/*.svs')

for slide_path in slides:
    slide_name = Path(slide_path).stem
    results = predictor.predict(
        wsi_path=slide_path,
        output_dir=f'results/{slide_name}',
        branch='he_nuclei',
        save_geojson=True
    )
    print(f"{slide_name}: {results['num_detections']} nuclei")
```

---

## 🔧 Model Checkpoints

Download pre-trained models:

| Model | Size | Modality | Download |
|-------|------|----------|----------|
| VitaminPFlex | Large | H&E or MIF | [Link](#) |
| VitaminPFlex | Base | H&E or MIF | [Link](#) |
| VitaminPDual | Base | H&E + MIF | [Link](#) |

Place checkpoints in `checkpoints/` folder.

---

## 🤔 Troubleshooting

### **"Out of memory" error**
```python
predictor = WSIPredictor(
    model=model,
    patch_size=512,
    overlap=32,  # Reduce from 64
    mixed_precision=True  # Enable FP16
)
```

### **No MPP in metadata**
```python
results = predictor.predict(
    wsi_path='image.png',
    mpp_override=0.25,  # Manually specify
    branch='he_nuclei'
)
```

### **Too many false positives**
```python
results = predictor.predict(
    wsi_path='slide.svs',
    detection_threshold=0.7,  # Increase from 0.5
    min_area_um=5.0,         # Filter small detections
    branch='he_nuclei'
)
```

---

## 📚 Citation

If you use VitaminP in your research, please cite:
```bibtex
@article{vitaminp2025,
  title={VitaminP: Robust Cell Segmentation for H&E and Multiplex IF},
  author={Your Name},
  journal={arXiv},
  year={2025}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

## 🙋 Support

- 🐛 **Issues:** [GitHub Issues](https://github.com/yourusername/vitaminp/issues)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/yourusername/vitaminp/discussions)
- 📧 **Email:** your.email@institution.edu

---

**Made with ❤️ for the computational pathology community**