# cybulde-model: End-to-End Cyberbullying Detection

An end-to-end machine learning project for detecting cyberbullying in text, built with production ML engineering practices. This repo combines the **data preparation pipeline** (`cybulde-data-preparation`) and the **model training pipeline** into a single codebase.

---

## Architecture Overview

The system is composed of two major distributed pipelines, both running on Google Cloud Platform (GCP):

### Data Preparation Pipeline
Raw datasets from multiple sources → distributed preprocessing VMs → cleaned, versioned data in GCS

![Data Preprocessing Architecture](cybulde_artifacts/478489890-1c7909ca-1fa8-4ffe-b132-b4c749264ea0.png)

### Model Training Pipeline
Processed data → distributed training across GCP VMs → tracked experiments in MLflow

![Model Training Architecture](cybulde_artifacts/478275008-92d1fc3a-2ebf-4190-9199-441955042240.png)

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Model framework | PyTorch Lightning |
| Configuration | Hydra (structured configs) |
| Dependency management | Poetry |
| Containerization | Docker + docker-compose |
| Cloud infrastructure | GCP (Compute Engine VMs, Cloud SQL, GCS) |
| Experiment tracking | MLflow |
| Data processing | Python (distributed across VMs) |
| Package tooling | pyproject.toml + setup.cfg |

---

## Repository Structure

```
cybulde-model/
│
├── cybulde/                    # Main Python package
│   ├── config/                 # Hydra structured config definitions
│   │   ├── config.py           # Root config dataclass (CybuldeConfig)
│   │   ├── training/           # Training hyperparameter configs
│   │   ├── model/              # Model architecture configs
│   │   ├── data/               # Data loading configs
│   │   └── gcp/                # GCP infrastructure configs
│   │
│   ├── data/                   # Data pipeline code
│   │   ├── datasets.py         # PyTorch Dataset classes
│   │   ├── data_modules.py     # PyTorch Lightning DataModules
│   │   └── processing/         # Text preprocessing utilities
│   │
│   ├── models/                 # Model definitions
│   │   ├── lightning_modules/  # PyTorch Lightning LightningModule subclasses
│   │   └── architectures/      # Raw model architectures (e.g., transformer heads)
│   │
│   ├── training/               # Training entrypoints and pipelines
│   │   └── train.py            # Main training script (decorated with @hydra.main)
│   │
│   └── utils/                  # Utilities
│       ├── gcp.py              # GCP helpers (GCS I/O, VM coordination)
│       └── io.py               # File I/O helpers
│
├── scripts/                    # Shell scripts for GCP infrastructure
│   └── deploy-etcd-server.sh   # Deploy etcd coordination server on GCE
│
├── docker/                     # Dockerfiles
│   ├── Dockerfile              # Production image
│   └── Dockerfile.dev          # Development image
│
├── .envs/                      # Environment variable files (not committed to git)
│   ├── .postgres               # PostgreSQL / MLflow backend config
│   ├── .mlflow-common          # Shared MLflow settings
│   ├── .mlflow-dev             # Dev MLflow settings
│   └── .infrastructure         # GCP project, registry, VM settings
│
├── mlruns/                     # MLflow experiment tracking data (local)
│
├── Makefile                    # Developer workflow commands
├── docker-compose.yaml         # Docker service definitions
├── pyproject.toml              # Project metadata + Poetry dependencies
├── setup.cfg                   # Additional tool configuration (linting, etc.)
└── poetry.lock                 # Locked dependency versions
```

---

## The Config System

This project uses **Hydra structured configs** — arguably its most complex and powerful component. Understanding it is key to working with the codebase.

### How It Works

Rather than plain YAML files, configuration is defined as **Python dataclasses** in `cybulde/config/config.py`. This gives you type checking, IDE autocompletion, and validation at startup.

```python
# cybulde/config/config.py (simplified)
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

@dataclass
class ModelConfig:
    name: str = "distilbert-base-uncased"
    num_labels: int = 2
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    max_epochs: int = 10
    learning_rate: float = 2e-5
    batch_size: int = 32
    precision: int = 16          # FP16 for faster training

@dataclass
class GcpConfig:
    project_id: str = "my-project"
    bucket_name: str = "my-bucket"
    data_path: str = "data/processed"

@dataclass
class CybuldeConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    gcp: GcpConfig = field(default_factory=GcpConfig)

# Register with Hydra's ConfigStore
cs = ConfigStore.instance()
cs.store(name="cybulde_config", node=CybuldeConfig)
```

### Overriding Config Values

Config overrides are passed via the `OVERRIDES` variable:

```bash
# Override individual values
make local-run-tasks OVERRIDES="training.max_epochs=20 training.learning_rate=1e-5"

# Run a hyperparameter sweep
make local-run-tasks OVERRIDES="-m training.learning_rate=1e-5,2e-5,3e-5"
```

### Instantiating Objects with Hydra

One of the most powerful patterns is using `hydra.utils.instantiate()` to create Python objects directly from config:

```python
# In config
@dataclass
class OptimizerConfig:
    _target_: str = "torch.optim.AdamW"
    lr: float = 2e-5
    weight_decay: float = 0.01

# In training code
optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
# This calls: torch.optim.AdamW(params=..., lr=2e-5, weight_decay=0.01)
```

This means you can swap optimizers, schedulers, or even model architectures entirely through config — no code changes required.

---

## Data Pipeline (from cybulde-data-preparation)

The data preparation codebase handles:

1. **Ingestion** — Loading raw cyberbullying text datasets from multiple sources (Kaggle datasets, Twitter data, etc.)
2. **Preprocessing** — Text cleaning, tokenization, label normalization
3. **Distributed execution** — Processing runs across multiple GCP VMs in parallel
4. **Output** — Processed datasets written to GCS, metadata written to Cloud SQL (PostgreSQL)

The `notebooks/` directory in `cybulde-data-preparation` contains exploratory analysis of the datasets.

Here's the distributed preprocessing running live on GCP:

![GCP preprocessing 1](cybulde_artifacts/478271350-f7d72b56-188f-4d82-84e3-be9deb131f1a.png)
![GCP preprocessing 2](cybulde_artifacts/478271446-9d718eb5-22ed-45a3-bf9a-27e2d52f2f16.png)
![GCP preprocessing 3](cybulde_artifacts/478271867-5b78648b-17a4-4d98-af30-751ef2651ebc.png)

---

## Model Architecture

The model is a transformer-based text classifier (e.g., DistilBERT fine-tuned for binary/multi-class cyberbullying detection). The PyTorch Lightning `LightningModule` in `cybulde/models/lightning_modules/` wraps the model and handles:

- Forward pass
- `training_step` / `validation_step` / `test_step`
- Optimizer and LR scheduler configuration
- MLflow metric logging

---

## Getting Started

### Prerequisites

- Docker and docker-compose
- GCP credentials configured (`gcloud auth application-default login`)
- A GCP project with Compute Engine, Cloud Storage, and Cloud SQL enabled
- Environment files populated (see below)

### Configure Environment

Copy the example env files and fill in your GCP details:

```bash
cp .envs/.infrastructure.example .envs/.infrastructure
cp .envs/.postgres.example .envs/.postgres
# etc. — edit each file with your project settings
```

### Run Locally (Development)

```bash
make local-run-tasks
```

This generates the final config and runs distributed training locally via `torchrun`. On a Mac M1:

```bash
make laptop-run-tasks
```

To pass config overrides:

```bash
make local-run-tasks OVERRIDES="training.max_epochs=2"
```

### Run on GCP

```bash
make run-tasks
```

This chains `generate-final-config` → `build` → `push` (to GCP Artifact Registry) → `launch_job_on_gcp.py` automatically. The Docker image is tagged with a unique UUID per run.

### Jupyter Notebook

```bash
make notebook
# Open http://localhost:8888
```

---

## Makefile Reference

Run `make help` to see all available targets with descriptions.

```bash
# Training
make local-run-tasks        # Run training locally (Docker + torchrun)
make laptop-run-tasks       # Run training locally on Mac M1
make run-tasks              # Build, push to GCP, and launch distributed training on GCP VMs

# Docker
make build                  # Build Docker image
make up                     # Start containers (detached)
make down                   # Stop containers
make exec-in                # Open interactive shell in container

# Code quality
make format-and-sort        # Format with black + isort
make lint                   # format-check + sort-check + flake8
make check-type-annotations # Run mypy
make test                   # Run pytest
make full-check             # lint + mypy + pytest with coverage

# Infrastructure
make deploy-etcd-server     # Deploy etcd coordination server on GCE
make mlflow-ssh-tunnel      # Open SSH tunnel to MLflow server on GCP VM

# Dependencies
make lock-dependencies      # Regenerate poetry.lock inside Docker
```

---

## Experiment Tracking

MLflow is used for experiment tracking. When running on GCP, the MLflow server runs on a dedicated VM. To access the UI, open an SSH tunnel:

```bash
make mlflow-ssh-tunnel
# Then open http://localhost:<PROD_MLFLOW_SERVER_PORT>
```

For local runs, MLflow logs to `mlruns/` and the UI can be accessed via:

```bash
docker-compose exec app-dev mlflow ui --host 0.0.0.0
# Open http://localhost:5000
```

Each run logs:
- Hyperparameters (from Hydra config)
- Training/validation loss and metrics per epoch
- Model checkpoints

---

## Data Preparation (Detailed)

For full details on the data preparation pipeline, see the companion repo: [cybulde-data-preparation](https://github.com/ajohnson114/cybulde-data-preparation).

The key datasets used include publicly available cyberbullying and hate speech text datasets. After preprocessing, data is stored in a consistent schema in GCS.

---

## Notes

- The `mlruns/` directory is committed for reference but would typically be excluded from version control in a team setting (use a remote MLflow tracking server instead).
- The `.envs/` directory contains secret environment variables and should **not** be committed to version control.
- The Makefile auto-detects whether you're on Apple M1 (`dev` profile) or CI/other (`ci` profile) and selects the appropriate Docker service accordingly.

---

## License

GPL-2.0 — see [LICENSE](LICENSE) for details.
