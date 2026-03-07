# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based Graph Neural Network (GNN) toolkit for training and evaluating models on molecular graph data. The system supports multiple operational modes: training, hyperparameter tuning, prediction, and fine-tuning.

## Setup & Installation

**Environment Setup:**
```bash
conda create -n gnn python=3.12
cd gnn
# For GPU
pip install -r envs/requirements-gpu.txt
# For CPU
pip install -r envs/requirements-cpu.txt
```

**Parameter Configuration:**
Configuration is managed via Hydra with YAML config groups in `configs/`.
Override settings on the command line or by editing the YAML files under `configs/`.

## Running the System

**All commands should be run from the `gnn/` directory.**

Training, hyperparameter tuning, and prediction all use the same entry point:
```bash
python main.py
```

The behavior is determined by the `mode` setting in the config:
- `training`: Train a model with fixed parameters
- `hparam_tuning`: Optimize hyperparameters using Optuna
- `prediction`: Generate predictions using a pre-trained model
- `fine-tuning`: Continue training a pre-trained model on new data

To run in the background with logging:
```bash
nohup python main.py > recording.log 2>&1 &
```

## High-Level Architecture

### Data Flow Pipeline

1. **Data Processing** (`data/data_processing.py`):
   - Loads molecular structures from SDF files
   - Processes node/edge/graph attributes
   - Creates PyTorch Geometric Data objects
   - Handles data normalization (stored in `norm_dict` for later use)
   - Splits data into train/validation/test sets

2. **Graph Dataset** (`data/graph_dataset.py`):
   - Inherits from PyTorch Geometric's InMemoryDataset
   - Applies feature transformations
   - Generates default attributes (element type, aromatic flags, bond types, etc.)
   - Supports custom node, edge, and graph-level attributes

3. **Model Architecture** (`models/`):
   - **Backbone + Head** architecture: backbone handles message passing, head handles readout
   - Backbones: MPNN, GCN, GAT, GIN, SchNet, DimeNet, PaiNN, MACE, GPS, Transformer
   - Heads: graph-level (pooling → MLP) and node-level
   - Configured via `configs/model/` YAML groups

4. **Training Loop** (`utils/train.py`, `main.py`):
   - Single function `train()` for one epoch of training with gradient accumulation
   - Single function `validate()` for validation without gradient updates
   - Supports multiple loss functions: MAE, MSE, Cosine
   - Evaluation metrics: MAE, MSE, RMSD, R², AARD, Cosine Similarity
   - Early stopping based on validation loss
   - Learning rate scheduling via ReduceLROnPlateau

5. **Model Management**:
   - **Saving** (`utils/save_model.py`): Tracks best model by loss, saves checkpoints at regular intervals
   - **Loading** (`utils/gen_model.py`): Generates model from dataset and parameters, loads pre-trained weights
   - **Normalization**: `norm_dict['y']` returns a `(mean, std)` tuple used for denormalization during evaluation

### Key Module Relationships

- **FileProcessing** (`utils/file_processing.py`): Creates directory structures, manages loggers, tracks error metrics
- **Evaluation** (`utils/evaluation.py`): Computes predictions and denormalizes using `norm_dict`
- **Visualization** (`utils/visualization.py`): Generates scatter plots, histograms, bar charts, and correlation heatmaps
- **Optuna Setup** (`utils/optuna_setup.py`): Configures hyperparameter optimization trials

### Data Format & Conventions

**Tensor Naming:**
- Target/Prediction shapes depend on `target_type`:
  - `graph`: Shape `[num_graphs, num_targets]`
  - `node`: Shape `[num_nodes, num_targets]`
  - `edge`: Shape `[num_edges, num_targets]`
- Normalization applied per-attribute via `norm_dict[attr]` → `(mean, std)` tuple

**Batch Processing:**
- Data loader returns PyTorch Geometric Data objects with batch information
- `batch` attribute indicates which nodes/edges belong to which graph
- Graph-level attributes concatenated before final linear layer

**File Organization:**
```
gnn/
├── configs/              # Hydra configuration files and schemas
├── data/                 # Dataset loading and transformation
├── models/               # Model definitions (backbones + heads)
├── utils/                # Training, evaluation, utilities
├── envs/                 # Requirements files
├── docs/                 # Documentation
├── Recording/            # Output directory (unified)
│   ├── Training_Recording/        # Training results
│   ├── HPTuning_Recording/        # Hyperparameter tuning results
│   └── Prediction_Recording/      # Predictions
└── main.py               # Entry point
```

## Important Implementation Details

### Denormalization & Evaluation

When evaluating predictions, targets must be denormalized using the normalization dictionary stored during data processing:
- Stored in `norm_dict['y']` as a `(mean, std)` tuple
- Used by `Evaluation` class to compute metrics on original scale
- Critical for multi-target prediction: normalization is computed per-column

### GPU Memory Management

- Set `GPU_memo_frac` in parameters to control per-process GPU memory fraction (0-1)
- CUBLAS workspace config: `CUBLAS_WORKSPACE_CONFIG=:4096:8` (set at module import)
- PYTHONHASHSEED set to 0 for reproducibility

### Optuna Integration

- Hyperparameter tuning via `optuna` library
- Configurable samplers (e.g., TPESampler) and pruners (e.g., MedianPruner)
- Results stored in SQLite database for resumable studies
- Each trial gets its own output directory with separate model and logs

## Common Development Tasks

**Adding a new metric:**
- Add method to `utils/metrics.py` Metrics class, add to `criteria_list` in parameters

**Adding a new model type:**
- Create backbone class in `models/backbones/`, register in `models/factory.py`

**Modifying data processing:**
- Edit `data/graph_dataset.py` for attribute generation or `data/data_processing.py` for loading/normalization

**Debugging training:**
- Check logs in `Recording/Training_Recording/<jobtype>/<TIME>/training_<TIME>.log`
- Monitor GPU memory in `gpu_monitor.log`
- Scatter plots in `Plot/` directory show prediction quality
- TensorBoard logs in `TensorBoard/` directory

## Code Style & Conventions

- Type hints use `dict` instead of `Dict` (Python 3.10+ style)
- Configuration via Hydra YAML groups in `configs/`
- Logging via Python's `logging` module to both file and console
- Model checkpoints include optimizer state for resumable training
