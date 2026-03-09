# GNN

A PyTorch-based Graph Neural Network toolkit for molecular property prediction. Supports multiple GNN architectures, graph/node/edge-level prediction, hyperparameter tuning, and fine-tuning.

## Quick Start

### 1. Clone

```bash
git clone git@github.com:hdj020402/GNN.git
cd gnn
```

### 2. Install

```bash
conda create -n gnn python=3.12
conda activate gnn

# GPU
pip install -r envs/requirements-gpu.txt

# CPU
pip install -r envs/requirements-cpu.txt
```

> **Note:** PyTorch and PyG extension packages (`torch-cluster`) may require special index URLs. See comments in the requirements files for details.

### 3. Configure

Configuration is managed via [Hydra](https://hydra.cc/) with YAML config groups under `configs/`. Key config files:

| Config Group | File | Description |
|---|---|---|
| Root | `configs/config.yaml` | Mode, seed, GPU settings |
| Data | `configs/data/default.yaml` | Dataset paths, features, split method |
| Backbone | `configs/model/backbone/<name>.yaml` | GNN backbone (mpnn, gcn, gat, gin, etc.) |
| Head | `configs/model/head/<name>.yaml` | Prediction head (graph, node) |
| Training | `configs/training/default.yaml` | Loss, optimizer, scheduler, epochs |
| Output | `configs/output/default.yaml` | Job type naming |
| Optuna | `configs/optuna/default.yaml` | Hyperparameter tuning settings |

Override any parameter on the command line:

```bash
python main.py mode=training training.lr=0.0005 model/backbone=gat training.epoch_num=500
```

### 4. Run

All modes use the same entry point:

```bash
python main.py               # default: training mode
python main.py mode=hparam_tuning
python main.py mode=prediction pretrained_model=path/to/model.pth
python main.py mode=fine-tuning pretrained_model=path/to/model.pth
```

Background execution:

```bash
nohup python main.py > recording.log 2>&1 &
```

### 5. Results

All outputs are organized under `Recording/`:

```
Recording/
├── Training_Recording/<jobtype>/<TIME>/
│   ├── Model/
│   │   ├── checkpoint/           # Periodic checkpoints
│   │   └── best_model_*.pth      # Best model by val metric
│   ├── Plot/                     # Scatter plots & training curves
│   ├── TensorBoard/              # TensorBoard logs
│   ├── training_*.log            # Training log
│   └── config.yaml               # Config snapshot
├── HPTuning_Recording/<jobtype>/<TIME>/
│   ├── Trial_000/ ... Trial_N/   # Per-trial results
│   ├── hptuning_*.db             # Optuna study database
│   └── hptuning_*.log
└── Prediction_Recording/<jobtype>/<TIME>/
    ├── Data/                     # Predictions & targets (.pt)
    ├── Plot/                     # Scatter plots
    └── prediction_*.log
```

## Supported Models

### Backbones

| Backbone | Type | Key Features |
|---|---|---|
| MPNN | Message-passing | NNConv + GRU, edge features |
| GCN | Message-passing | Standard graph convolution |
| GAT | Attention | Multi-head attention |
| GIN | Message-passing | Maximally expressive for WL test |
| Transformer | Attention | Multi-head + edge features |
| GPS | Hybrid | Local MPNN + global attention |
| SchNet | Equivariant | Continuous filters, radial basis |
| PaiNN | Equivariant | Scalar + vector features |
| DimeNet++ | Equivariant | Directional message passing |
| MACE | Equivariant | Multi-body interactions, ACE basis |

### Prediction Heads

| Head | Target Type | Readout Options |
|---|---|---|
| Graph | graph, vector | Set2Set, mean, max, sum |
| Node | node | Per-node MLP |

## Input Data Format

- **SDF file** (required): Molecular 3D structures
- **Graph attribute CSV** (required): Targets and graph-level features
- **Node/Edge attribute JSON or PKL** (optional): Custom per-atom/per-bond features
- **Vector target file** (optional): For vector-type targets

See `configs/data/default.yaml` for all data options and `docs/tutorial.md` for detailed format specifications.

## Documentation

See [`docs/tutorial.md`](docs/tutorial.md) for a comprehensive guide covering data formats, model architecture, configuration details, and training workflow.
