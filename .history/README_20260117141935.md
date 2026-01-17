# HEXIF: Histology to Multiplex Immunofluorescence Translation

This repository contains the official implementation of the HEXIF model, a deep learning framework for synthesizing multiplex immunofluorescence (ORION/CODEX) images directly from standard H&E stained histology slides.

## Installation

1. Clone the repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Note:** `pyvips` and `valis` require system-level dependencies (e.g., libvips). Please refer to their respective documentation for installation instructions if pip install fails.

## Usage

### 1. Inference (Prediction)

To generate virtual multiplex images from an H&E input:

```bash
python predict.py --input path/to/slide.svs --output_dir results/ --model_path path/to/model.pth --scaler_path path/to/scaler.json
```

**Arguments:**
- `--input`: Path to a single H&E image file (e.g., .tif, .svs, .png) or a directory of images.
- `--output_dir`: Directory where predicted images (.npy) and diagnostic plots (.png) will be saved.
- `--model_path`: Path to the trained model checkpoint.
- `--scaler_path`: Path to the scaler configuration JSON file (generated during training).
- `--no_tissue_detect`: (Optional) Flag to disable automatic tissue core detection for Whole Slide Images.

**Output:**
For each detected tissue core or input image, the script generates:
- `*_pred.npy`: Raw predicted 20-channel array.
- `*_diagnostic.png`: Visualization of the input H&E and predicted channels.

### 2. Data Preparation

To prepare data for training (requires paired H&E and Orion/CODEX images):

**Step 1: Registration (Optional)**
If your H&E and Orion images are not aligned:

```bash
python preprocess.py register --he_slide path/to/he.ome.tiff --orion_slide path/to/orion.ome.tiff --output_dir data/registered
```

**Step 2: Core Extraction**
Extract aligned tissue cores into training patches (.npy format):

```bash
python preprocess.py extract --he_slide path/to/registered_he.ome.tiff --orion_slide path/to/registered_orion.ome.tiff --output_dir core_patches_npy
```

### 3. Training

To retrain the model on new data:

```bash
torchrun --nproc_per_node=4 train.py --pairs_dir core_patches_npy --output_dir runs/new_experiment
```

**Key Arguments:**
- `--pairs_dir`: Directory containing paired `_HE.npy` and `_ORION.npy` files.
- `--epochs`: Number of training epochs (default: 80).
- `--loss_type`: Loss function (`l1` or `l2`).
- `--w_center`, `--w_msssim`, `--w_cov`: Weights for different loss components.

## Repository Structure

- `hexif/`: Python package containing model architecture, data loading, and processing logic.
- `predict.py`: Entry point for inference and diagnostics.
- `preprocess.py`: Tools for slide registration and patch extraction.
- `train.py`: Script for training the model (supports distributed training).
- `requirements.txt`: Python dependencies.
- `archive/`: Deprecated scripts and files.

## Model Architecture

The model uses a SwinUNet architecture (Swin Transformer encoder + U-Net decoder) to map H&E patches to high-dimensional multiplex channels. It employs a combined loss function including pixel-wise reconstruction, structural similarity (MS-SSIM), and channel-wise coverage constraints.
