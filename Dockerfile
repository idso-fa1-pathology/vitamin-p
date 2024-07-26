FROM tensorflow/tensorflow:latest-gpu

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV HDF5_USE_FILE_LOCKING=FALSE
ENV NUMBA_CACHE_DIR=/tmp

# Install system packages with retry logic
RUN for i in {1..5}; do \
    (echo "Attempt $i: Updating and installing packages..." && \
    apt-get update -o Acquire::https::Timeout="60" -o Acquire::http::Timeout="60" -o Acquire::Retries="3" && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        curl \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "Packages installed successfully" && break) || \
    (echo "Attempt $i failed. Retrying in 30 seconds..." && sleep 30); \
    done

# Install Python packages
RUN for i in {1..3}; do \
    (echo "Attempt $i: Installing Python packages..." && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        gpustat==0.6.0 \
        setuptools==61.2.0 \
        pytz==2021.1 \
        joblib==1.2.0 \
        tqdm==4.64.0 \
        docopt==0.6.2 \
        ipython==8.10.0 \
        jupyterlab==3.6.1 \
        notebook==6.4.11 \
        traitlets==5.9.0 \
        chardet==5.0.0 \
        nbconvert==7.8.0 \
        Pillow==10.0.0 \
        pandas==2.1.2 \
        matplotlib==3.7.2 \
        seaborn==0.13.0 \
        pycm==3.5 \
        deepdish==0.3.7 \
        opencv-python==4.8.0.74 \
        scikit-image==0.22.0 \
        imgaug==0.4.0 \
        scikit-learn==1.3.2 \
        xgboost==2.0.3 \
        statsmodels==0.13.5 \
        tensorflow==2.15.0 \
        torch \
        torchvision \
        transformers==4.36.2 && \
    echo "Python packages installed successfully" && break) || \
    (echo "Attempt $i failed. Retrying in 30 seconds..." && sleep 30); \
    done

# Configure folder permissions
RUN mkdir -p /tensorflow_datasets /.dgl /.local /.cache /Data /App && \
    chmod -R 777 /tensorflow_datasets /.dgl /.local /.cache /Data

# Set working directory
WORKDIR /App

CMD ["/bin/bash"]