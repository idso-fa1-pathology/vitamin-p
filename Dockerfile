FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

# Set web proxy for MD Anderson network
ENV http_proxy=http://1mcwebproxy01.mdanderson.edu:3128
ENV https_proxy=http://1mcwebproxy01.mdanderson.edu:3128
ENV HTTP_PROXY=http://1mcwebproxy01.mdanderson.edu:3128
ENV HTTPS_PROXY=http://1mcwebproxy01.mdanderson.edu:3128

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV HDF5_USE_FILE_LOCKING=FALSE
ENV NUMBA_CACHE_DIR=/tmp
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system libraries
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
    sudo \
    curl \
    wget \
    htop \
    git \
    vim \
    ca-certificates \
    python3-openslide \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python commands
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install base Python packages
RUN pip install --no-cache-dir \
    gpustat==0.6.0 \
    setuptools==61.2.0 \
    pytz==2021.1 \
    joblib==1.2.0 \
    tqdm==4.64.0 \
    docopt==0.6.2

# Install Jupyter and notebook tools
RUN pip install --no-cache-dir \
    ipython==8.10.0 \
    jupyterlab==3.6.1 \
    notebook==6.4.11 \
    traitlets==5.9.0 \
    chardet==5.0.0 \
    nbconvert==7.8.0

# Install image processing libraries (adjusted versions for Python 3.8)
RUN pip install --no-cache-dir \
    openslide-python==1.3.1 \
    Pillow==10.0.0 \
    opencv-python==4.8.0.74 \
    scikit-image==0.21.0 \
    tifffile==2023.7.10 \
    imagecodecs==2023.3.16

# Install data science packages (adjusted versions for Python 3.8)
RUN pip install --no-cache-dir \
    numpy==1.24.4 \
    pandas==2.0.3 \
    matplotlib==3.7.2 \
    seaborn==0.13.0 \
    scikit-learn==1.3.2 \
    geopandas==0.13.2 \
    scipy==1.10.1

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 🔥 Install CuPy for GPU-accelerated post-processing (CUDA 12.1)
RUN pip install --no-cache-dir cupy-cuda12x

RUN pip install --no-cache-dir \
    wandb \
    timm

# Install additional ML/DL packages
RUN pip install --no-cache-dir \
    clip-anytorch==2.6.0 \
    tensorboard==2.14.0 \
    albumentations \
    PyYAML \
    "zarr<3"

# Configure folder permissions
RUN mkdir -p /.dgl /.local /.cache /tmp && \
    chmod -R 777 /.dgl /.local /.cache /tmp

# Create Data folder
WORKDIR /Data
RUN chmod -R 777 /Data

# Create workspace and set permissions
WORKDIR /workspace
RUN chmod -R 777 /workspace

# Copy vitaminp package (main package with all modules)
COPY vitaminp/ /workspace/vitaminp/

# Copy scripts (training and inference scripts)
COPY scripts/ /workspace/scripts/

# Copy dataset folder if needed
COPY dataset/ /workspace/dataset/

# Copy configuration files
COPY configs/ /workspace/configs/

# Copy additional files
COPY setup.py /workspace/
COPY requirements.txt /workspace/
COPY README.md /workspace/
COPY LICENSE /workspace/

# Create necessary directories
RUN mkdir -p /workspace/checkpoints \
             /workspace/cache \
             /workspace/output \
             /workspace/inference_results \
             /workspace/results \
             /workspace/test_images && \
    chmod -R 777 /workspace

# Set Python path
ENV PYTHONPATH=/workspace

# Default working directory
WORKDIR /workspace

CMD ["/bin/bash"]