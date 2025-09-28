# ML-based Clustering and Dimensionality Reduction for Magnetic Field Simulations

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Sassy-Buoy/hyper-param-optim-4-vector-field-clustering/HEAD?urlpath=%2Fdoc%2Ftree%2Ftest.ipynb)

This project implements autoencoder-based dimensionality reduction and clustering for vector field data. The system trains autoencoder and variational autoencoders to learn meaningful representations of magnetic field simulations for clustering analysis. Hyperparameter optimization is performed using Optuna.

## Micromagnetic Simulations

- Simulations of a FeGe (iron germanide) disc (diameter = 160 nm, thickness = 10 nm) under varying external magnetic field (0 to 1.2 T). 
- The physical energy contributions used include exchange, Dzyaloshinskii–Moriya interaction (DMI), magnetostatic energy, Zeeman energy.
- Discretized with a mesh of cubic cells of side 2 nm (leading to an effective grid of size ~ 80×80 in-plane × 3 components) so that each field configuration is a 3-component vector per cell. 
- Many equilibrium states per applied field by varying initial conditions (to explore possible metastable states), obtaining 3010 equilibrium states to classify.

## Project Overview

The project focuses on:
- **Vector Field Processing**: Loading and preprocessing magnetic field simulation data
- **Autoencoder Training**: Implementing both vanilla and variational autoencoders for dimensionality reduction
- **Clustering**: Analyzing latent representations for clustering patterns
- **Hyperparameter Optimization**: Using Optuna for automated hyperparameter tuning
- **Visualization**: Plotting training metrics, reconstructions, and UMAP embeddings

## Project Structure

### Data Directory (**`data/`**)

- **`data-generation-scripts/`** : Scripts for generating magnetic field simulations. See another repo for details.
- **`sims/`** : Raw simulation files in OOMMF format.
- **`parameters_dict.json`**: Simulation parameters for each data sample
- **`simulation_file_paths.json`**: File paths to original simulation files
- **`sim_arr_tensor.pt`**: Preprocessed tensor data containing magnetic field simulations. Not included due to size constraints.
- **`labels.npy`**: Ground truth clustering labels for evaluation
- **`load_data.py`**: Data loading utilities for processing raw simulation files.
- **`data/field_images/`**: Directory containing magnetic field visualization images in .png format. Useful for inspecting the clustering quality.

### Models Directory (**`models/`**)

- **`__init__.py`**: Package initialization, exports main classes
- **`auto_encoder.py`**: Implementation of autoencoder and variational autoencoder architectures
- **`lit_model.py`**: PyTorch Lightning module for training, validation, and data handling

### Plotting Directory (**`plot/`**)

- **`__init__.py`**: Package initialization
- **`plotting.py`**: Magnetic field visualization utilities with matplotlib and plotly
- **`plot_recon.py`**: Reconstruction quality visualization
- **`plot_umap.py`**: UMAP embedding visualization
- **`training_log.py`**: Training metrics plotting

### 

### Hyperparameter Optimization (**`z_hyperopt/`**)

- **`run.py`**: Main Optuna optimization script
- **`search_space.py`**: Defines hyperparameter search spaces
- **`cross_validation.py`**: Cross-validation utilities
- **`deep_embedding.py`**: Deep embedding model implementations
- **`optuna.db`**: Optuna study database
- **`run.sh`**, **`run_parallel.sh`**: Scripts for running optimization jobs

### Training Logs (**`lightning_logs/`**)

- **`version_*/`**: PyTorch Lightning training logs, checkpoints, and metrics for different training runs. Model checkpoints are automatically saved based on validation performance.

### Root Files

- **`run.py`**: Main training script that loads configuration from `config.yaml` and trains the autoencoder model using PyTorch Lightning
- **`config.yaml`**: Configuration file defining model architecture, hyperparameters, and training settings
- **`cluster_acc.py`**: Clustering evaluation metrics including purity score and adjusted Rand index
- **`run.sh`**: Shell script for running training jobs (likely for HPC environments)
- **`test.ipynb`**: Jupyter notebook for testing and experimentation

## Key Components

### Autoencoder Architectures

The project implements two types of autoencoders:

1. **Vanilla Autoencoder**: Standard encoder-decoder architecture for dimensionality reduction
2. **Variational Autoencoder (VAE)**: Probabilistic autoencoder with KL divergence regularization

Both models are configured through the `config.yaml` file with customizable:
- Convolutional layers (channels, kernel sizes, strides)
- Fully connected layers
- Activation functions
- Latent space dimensions

### Training Pipeline

The training uses PyTorch Lightning with:
- **Mixed precision training** (16-bit) for efficiency
- **Early stopping** based on validation loss
- **Model checkpointing** to save best models
- **Distributed training** support for multi-GPU setups
- **Gradient accumulation** for effective larger batch sizes

### Data Processing

- Loads magnetic field simulation data as 3D tensors
- Applies train/validation/test splits (60%/16%/20%)
- Handles batch processing with configurable batch sizes
- Supports data augmentation and normalization

### Clustering Evaluation

- **Purity Score**: Measures cluster homogeneity
- **Adjusted Rand Index**: Measures clustering similarity to ground truth
- Evaluates clustering performance on learned latent representations

## Usage

### Basic Training

```bash
python run.py
```

This loads configuration from `config.yaml` and trains the model specified therein.

### Hyperparameter Optimization

```bash
cd z_hyperopt
python run.py
```

Runs Optuna optimization to find best hyperparameters for clustering performance.

### Configuration

Edit `config.yaml` to modify:
- Model type (`"vanilla"` or `"variational"`)
- Architecture (layer configurations)
- Training parameters (learning rate, batch size, etc.)
- VAE-specific settings (KL divergence threshold)

## Output

The training produces:
- Trained model weights
- Best model checkpoints
- Latent space representations for each epoch
- Training metrics