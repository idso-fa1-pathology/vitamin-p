FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# set web proxy
ENV http_proxy http://1mcwebproxy01.mdanderson.edu:3128
ENV https_proxy http://1mcwebproxy01.mdanderson.edu:3128
ENV HTTP_PROXY http://1mcwebproxy01.mdanderson.edu:3128
ENV HTTPS_PROXY http://1mcwebproxy01.mdanderson.edu:3128

# set environment variables
ENV DEBIAN_FRONTEND noninteractive
ENV HDF5_USE_FILE_LOCKING FALSE
ENV NUMBA_CACHE_DIR /tmp

# install libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential libgl1-mesa-glx libglib2.0-0 libgeos-dev libvips-tools \
  sudo curl wget htop vim ca-certificates python3-openslide python3-pip python3-dev \
  && rm -rf /var/lib/apt/lists/*

# install python packages
RUN pip install gpustat==0.6.0 setuptools==61.2.0 pytz==2023.3 joblib==1.2.0 future==0.18.2 docopt==0.6.2 
RUN pip install tqdm==4.66.1 mako==1.2.4 einops==0.6.1 geojson==3.0.0 h5py==3.9.0 kaleido==0.2.1 natsort==8.4.0
RUN pip install ipython==8.10.0 jupyterlab==3.6.1 notebook==6.4.11 traitlets==5.9.0 chardet==5.0.0 nbconvert==7.8.0 pyjwt==2.6.0
RUN pip install pandas==2.1.2 pandarallel==1.6.5 matplotlib==3.7.2 seaborn==0.13.0 pycm==3.5 deepdish==0.3.7 openpyxl==3.1.2 XlsxWriter==3.2.0
RUN pip install numpy==1.23.0 scikit-learn==1.3.2 xgboost==2.0.3 statsmodels==0.13.5 lifelines==0.27.8 rasterio==1.3.5.post1 ujson==5.8.0
RUN pip install openslide-python==1.3.1 Pillow==10.0.0 opencv-python==4.8.0.74 scikit-image==0.22.0 imgaug==0.4.0 mahotas==1.4.12 Shapely==1.8.5.post1
RUN pip install imagecodecs==2024.6.1 tifffile==2024.5.22 pydantic==1.10.4 imageio==2.31.1 csbdeep==0.7.4 stardist==0.8.5 albumentations==1.3.0 histolab==0.6.0
RUN pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 torchinfo==1.8.0 torchmetrics==1.4.0 
RUN pip install wandb==0.15.4 schema==0.7.5 tabulate==0.9.0
RUN pip install cucim-cu12==24.2.0 cupy-cuda12x==12.0.0 
RUN pip install csbdeep==0.7.4 stardist==0.8.5

# configure folder permission
WORKDIR /.dgl
RUN chmod -R 777 /.dgl
WORKDIR /.local
RUN chmod -R 777 /.local
WORKDIR /.cache
RUN chmod -R 777 /.cache

# set working directory 
WORKDIR /App
CMD ["/bin/bash"]
# CMD["/usr/bin/python3", "python_script.py"]
