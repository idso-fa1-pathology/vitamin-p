from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Core dependencies (always installed)
INSTALL_REQUIRES = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "numpy>=1.20.0",
    "scipy>=1.8.0",
    "Pillow>=9.0.0",
    "tqdm>=4.60.0",
    "matplotlib>=3.5.0",
    "pandas>=1.3.0",
    "scikit-learn>=0.23.0",
    "pydantic>=2.0.0",
    "PyYAML>=5.4.0",
    "requests>=2.25.0",
    "packaging>=21.0",
    "sympy>=1.10.0",
    "networkx>=2.4",
    "filelock>=3.0.0",
]

# Optional extras
EXTRAS_REQUIRE = {
    # pip install vitaminp[gpu]
    # Just a reminder — torch GPU is installed separately via pytorch.org
    "gpu": [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        # For CUDA, install torch manually from https://pytorch.org/get-started/locally/
    ],

    # pip install vitaminp[wsi]  — whole slide image support
    "wsi": [
        "openslide-python>=1.2.0",
        "tifffile>=2022.0.0",
        "shapely>=1.8.0",
        "pyarrow>=7.0.0",       # for parquet saving
        "geopandas>=0.10.0",    # for geojson saving
    ],

    # pip install vitaminp[gan]  — GAN / synthesis support
    "gan": [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
    ],

    # pip install vitaminp[dev]  — development & testing
    "dev": [
        "pytest>=6.0.0",
        "flake8>=4.0.0",
        "pre-commit>=2.0.0",
        "notebook>=6.0.0",
        "ipython>=7.0.0",
        "tensorboard>=2.10.0",
    ],

    # pip install vitaminp[full]  — everything
    "full": [
        "openslide-python>=1.2.0",
        "tifffile>=2022.0.0",
        "shapely>=1.8.0",
        "pyarrow>=7.0.0",
        "geopandas>=0.10.0",
        "pytest>=6.0.0",
        "flake8>=4.0.0",
        "notebook>=6.0.0",
        "ipython>=7.0.0",
        "tensorboard>=2.10.0",
    ],
}

setup(
    name="vitaminp",
    version="0.2.1",
    author="Yasin Shokrollahi",
    author_email="YShokrollahi@mdanderson.org",          # <-- update this
    description="Cell and nucleus segmentation for whole slide images (H&E and MIF)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/idso-fa1-pathology/VitaminP",  # <-- update this
    packages=find_packages(),
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "cell segmentation",
        "nucleus segmentation",
        "pathology",
        "whole slide image",
        "WSI",
        "H&E",
        "MIF",
        "deep learning",
        "pytorch",
    ],
)