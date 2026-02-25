# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based Graph Neural Network (GNN) toolkit for training and evaluating models on molecular graph data. The system supports multiple operational modes: training, hyperparameter tuning, feature filtration, prediction, and fine-tuning.

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
1. Create `model_parameters.yml` in the `gnn/` directory (copy from `model_parameters_example.yml`)
2. Optionally create `hparam_tuning.yml` for hyperparameter tuning mode
3. Optionally create `feature_filter.yml` for feature filtration

## Running the System

**All commands should be run from the `gnn/` directory.**

Training, hyperparameter tuning, and prediction all use the same entry point:
```bash
python main.py
```

The behavior is determined by the `mode` setting in `model_parameters.yml`:
- `training`: Train a model with fixed parameters
- `hparam_tuning`: Optimize hyperparameters using Optuna
- `feature_filtration`: Test different feature combinations
- `prediction`: Generate predictions using a pre-trained model
- `fine-tuning`: Continue training a pre-trained model on new data

To run in the background with logging:
```bash
nohup python main.py > <MODE>_Recording/recording.log 2>&1 &
```

Replace `<MODE>` with `Training`, `HPTuning`, or `Prediction` as appropriate.

## High-Level Architecture

### Data Flow Pipeline

1. **Data Processing** (`utils/data_processing.py`):
   - Loads molecular structures from SDF files
   - Processes node/edge/graph attributes
   - Creates PyTorch Geometric Data objects
   - Handles data normalization (stored in `norm_dict` for later use)
   - Splits data into train/validation/test sets

2. **Graph Dataset** (`datasets/graph_dataset.py`):
   - Inherits from PyTorch Geometric's InMemoryDataset
   - Applies feature transformations and filters
   - Generates default attributes (element type, aromatic flags, bond types, etc.)
   - Supports custom node, edge, and graph-level attributes

3. **Model Architecture** (`nets/`):
   - **Graph-level prediction**: Uses NNConv (message passing with edge features) + Set2Set pooling + GRU for temporal modeling
   - **Node-level prediction**: Uses NNConv + GRU layers
   - Configurable dimensions via `dim_conv` and `dim_linear` parameters
   - Message passing iterations controlled by `mp_times` parameter
   - Graph-level features can be concatenated with readout before final prediction

4. **Training Loop** (`utils/train.py`, `main.py`):
   - Single function `train()` for one epoch of training with gradient accumulation
   - Single function `validate()` for validation without gradient updates
   - Supports multiple loss functions: MAE, MSE
   - Evaluation metrics: MAE, MSE, RMSD, R², AARD, Cosine Similarity
   - Early stopping based on validation loss
   - Learning rate scheduling via ReduceLROnPlateau

5. **Model Management**:
   - **Saving** (`utils/save_model.py`): Tracks best model by loss, saves checkpoints at regular intervals
   - **Loading** (`utils/gen_model.py`): Generates model from dataset and parameters, loads pre-trained weights
   - **Normalization**: `norm_dict` contains per-target normalization factors (mean/std) for denormalization during evaluation

### Key Module Relationships

- **FileProcessing** (`utils/file_processing.py`): Creates directory structures, manages loggers, tracks error metrics
- **Evaluation** (`utils/evaluation.py`): Computes predictions and denormalizes using `norm_dict`
- **Post-processing** (`utils/post_processing.py`): Parses log files, extracts performance metrics
- **Plotting** (`utils/plot.py`, `utils/plot_model.py`): Generates loss curves and scatter plots
- **Optuna Setup** (`utils/optuna_setup.py`): Configures hyperparameter optimization trials

### Data Format & Conventions

**Tensor Naming:**
- Target/Prediction shapes depend on `target_type`:
  - `graph`: Shape `[num_graphs, num_targets]`
  - `node`: Shape `[num_nodes, num_targets]`
  - `edge`: Shape `[num_edges, num_targets]`
- Normalization applied per-target via `norm_dict['<target_name>']['mean']` and `['std']`

**Batch Processing:**
- Data loader returns PyTorch Geometric Data objects with batch information
- `batch` attribute indicates which nodes/edges belong to which graph
- Graph-level attributes concatenated before final linear layer

**File Organization:**
```
gnn/
├── datasets/          # Dataset loading and transformation
├── nets/             # Model definitions
├── utils/            # Training, evaluation, utilities
├── envs/             # Requirements files
├── docs/             # Documentation
├── model_parameters.yml          # Main configuration
├── hparam_tuning.yml            # Hyperparameter tuning config (optional)
├── feature_filter.yml           # Feature filtering config (optional)
├── Training_Recording/          # Output: training results
├── HPTuning_Recording/          # Output: hyperparameter tuning results
├── Prediction_Recording/        # Output: predictions
└── main.py           # Entry point
```

## Important Implementation Details

### Denormalization & Evaluation

When evaluating predictions, targets must be denormalized using the normalization dictionary stored during data processing:
- Stored in `norm_dict['<target_name>']['mean']` and `['std']`
- Used by `Evaluation` class to compute metrics on original scale
- Critical for multi-target prediction: each target has separate normalization

### GPU Memory Management

- Set `GPU_memo_frac` in parameters to control per-process GPU memory fraction (0-1)
- CUBLAS workspace config: `CUBLAS_WORKSPACE_CONFIG=:4096:8` (set at module import)
- PYTHONHASHSEED set to 0 for reproducibility

### Feature Filtering

Supports two approaches:
- `one_by_one`: Leave-one-out feature elimination - removes each feature sequentially and trains
- `file`: Manual feature combinations specified in `feature_filter.yml` (not yet fully implemented)

### Optuna Integration

- Hyperparameter tuning via `optuna` library
- Configurable samplers (e.g., TPESampler) and pruners (e.g., MedianPruner)
- Results stored in SQLite database for resumable studies
- Each trial gets its own output directory with separate model and logs

## Common Development Tasks

**Adding a new metric:**
- Add method to `calc_error.py` class, add to `criteria_list` in parameters

**Adding a new model type:**
- Create model class in `nets/`, update `gen_model.py` to instantiate based on parameters

**Modifying data processing:**
- Edit `datasets/graph_dataset.py` for attribute generation or `utils/data_processing.py` for loading/normalization

**Debugging training:**
- Check logs in `Training_Recording/<jobtype>/<TIME>/training_<TIME>.log`
- Monitor GPU memory in `gpu_monitor.log`
- Scatter plots in `Plot/` directory show prediction quality

## Code Style & Conventions

- Type hints use `dict` instead of `Dict` (Python 3.10+ style)
- Configuration files are YAML format
- Logging via Python's `logging` module to both file and console
- Model checkpoints include optimizer state for resumable training
