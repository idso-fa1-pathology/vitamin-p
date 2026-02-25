FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04


# ── Environment ───────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV HDF5_USE_FILE_LOCKING=FALSE
ENV NUMBA_CACHE_DIR=/tmp
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/workspace

# ── System libraries ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgeos-dev \
    libvips-tools \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libopenslide-dev \
    openslide-tools \
    sudo \
    curl \
    wget \
    htop \
    git \
    vim \
    ca-certificates \
    python3 \
    python3-pip \
    python3-dev \
    python3-openslide \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN python3 -m pip install --upgrade pip setuptools wheel

# ── PyTorch (CUDA 12.1) ───────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Core scientific stack ─────────────────────────────────────────────────────
# Pinned to <2 to avoid OpenCV/NumPy 2.x incompatibility
RUN pip install --no-cache-dir \
    "numpy<2" \
    pandas==2.1.4 \
    scipy==1.11.4 \
    scikit-learn==1.3.2 \
    scikit-image==0.22.0 \
    matplotlib==3.8.2 \
    seaborn==0.13.0

# ── Image processing ──────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    Pillow==10.2.0 \
    opencv-python==4.9.0.80 \
    tifffile==2024.1.30 \
    imagecodecs==2024.1.1 \
    openslide-python==1.3.1

# ── Geospatial / WSI ──────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    geopandas==0.14.3 \
    shapely==2.0.3 \
    fiona==1.9.5

# ── Parquet / Arrow (for save_parquet=True) ───────────────────────────────────
RUN pip install --no-cache-dir \
    pyarrow==15.0.0 \
    fastparquet==2024.2.0

# ── Deep learning extras ──────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    timm==0.9.16 \
    albumentations==1.3.1 \
    wandb \
    tensorboard==2.15.1 \
    huggingface-hub==0.20.3

# ── GPU accelerated post-processing ──────────────────────────────────────────
RUN pip install --no-cache-dir cupy-cuda12x

# ── Misc utilities ────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    gpustat \
    tqdm==4.66.1 \
    PyYAML==6.0.1 \
    pydantic==2.5.3 \
    "zarr<3" \
    joblib==1.3.2 \
    natsort \
    click \
    rich

# ── Jupyter (unpinned to avoid conflicts) ─────────────────────────────────────
RUN pip install --no-cache-dir \
    jupyterlab \
    notebook \
    ipython \
    ipywidgets \
    nbconvert

# ── Install vitaminp from PyPI ────────────────────────────────────────────────
RUN pip install --no-cache-dir vitaminp

# ── Download pretrained weights from HuggingFace ─────────────────────────────
# Weights are baked into the image — no download needed at runtime
RUN python3 -c "\
from huggingface_hub import hf_hub_download; \
import os; \
os.makedirs('/workspace/checkpoints', exist_ok=True); \
hf_hub_download(repo_id='yasinmdanderson/vitaminp-weights', filename='vitamin_p_dual.pth', local_dir='/workspace/checkpoints'); \
hf_hub_download(repo_id='yasinmdanderson/vitaminp-weights', filename='vitamin_p_flex.pth', local_dir='/workspace/checkpoints'); \
print('✅ Weights downloaded successfully') \
"

# Also copy weights to cache so vitaminp.load_model() finds them instantly
RUN python3 -c "\
import os, shutil; \
cache = os.path.expanduser('~/.cache/vitaminp'); \
os.makedirs(cache, exist_ok=True); \
shutil.copy('/workspace/checkpoints/vitamin_p_dual.pth', cache); \
shutil.copy('/workspace/checkpoints/vitamin_p_flex.pth', cache); \
print('✅ Weights cached for load_model()') \
"

# ── Directory structure ───────────────────────────────────────────────────────
RUN mkdir -p \
    /.dgl /.local /.cache /tmp \
    /workspace/checkpoints \
    /workspace/cache \
    /workspace/output \
    /workspace/results \
    /workspace/test_images \
    /Data && \
    chmod -R 777 /.dgl /.local /.cache /tmp /workspace /Data

# ── Copy source (for development use; pip install handles production) ─────────
COPY vitaminp/   /workspace/vitaminp/
COPY scripts/    /workspace/scripts/
COPY configs/    /workspace/configs/
COPY setup.py    /workspace/
COPY requirements.txt /workspace/
COPY README.md   /workspace/
COPY LICENSE     /workspace/

WORKDIR /workspace
CMD ["/bin/bash"]
