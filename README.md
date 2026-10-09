# GraphGuard: GNN Network Intrusion Detection & Counterfactual Containment

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.5+-3C2179.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GraphGuard** is a Graph Neural Network (GNN) intrusion detection and cost-aware automated containment platform. It models high-velocity NetFlow telemetry as dynamic multi-attributed communication graphs, detects sophisticated multi-stage cyber attacks at line rate using **E-GraphSAGE**, and executes the closed **CVC Loop** (*Counterfactual → Verify → Contain*) to compute the minimal disruption network cut protecting critical enterprise assets while preserving legitimate traffic.

---

## Key Capabilities

- **E-GraphSAGE Edge Classifier:** Edge-attributed GraphSAGE architecture that classifies network flows across 10 attack categories (Benign, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Backdoor, Analysis, Worms, Shellcode).
- **CVC Closed Loop (Counterfactual $\to$ Verify $\to$ Contain):** Formulates containment as a cost-penalized continuous optimization problem, discovering the minimal edge cut required to isolate threats without taking down business-critical services.
- **Cost & Criticality Aware:** Evaluates asset criticality (e.g. Domain Controllers, Gateways, DNS, Databases) and flow throughput costs to prevent collateral damage.
- **Open-Set / Zero-Day Detection:** Free-energy logit scoring detects out-of-distribution, novel, and zero-day attack patterns without model retraining.
- **Dry-Run & Graduated Enforcement:** Supports non-destructive testing via `DryRunEnforcer` and graded actions (`DROP`, `RATE_LIMIT`, `QUARANTINE_PORT`, `ISOLATE_HOST`).

---

## Architecture Overview

```
                      NetFlow Telemetry (v2/v3 / IPFIX)
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │    WindowBuilder (Graph)      │
                      │  IPs -> Nodes, Flows -> Edges │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │    E-GraphSAGE GNN Detector   │
                      │   Edge Scores & Free Energy   │
                      └───────────────┬───────────────┘
                                      │ (Intrusion Flagged)
                                      ▼
                      ┌───────────────────────────────┐
                      │    Counterfactual Cut Solver  │  ◄── Asset Criticality Config
                      │     (Min Disruption Cut M_e)  │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │     Enforcement Engine        │
                      │  (DryRun / Firewall / iptables│
                      └───────────────────────────────┘
```

---

## Quick Start (Run in 1 Minute)

A pre-trained model and authentic NetFlow sample dataset (`data/sample_nf_v2.csv`) are bundled right in the repository.

### 1. Clone & Install

```bash
# Clone repository
git clone https://github.com/nitinrohilla-05/GraphGaurd-Network-intrusion-detection-using-graph-neural-netwrok.git
cd GraphGaurd-Network-intrusion-detection-using-graph-neural-netwrok

# Create virtual environment
python -m venv .venv

# Activate environment:
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

*Alternatively, if using [uv](https://github.com/astral-sh/uv):*
```bash
uv venv --python 3.11 .venv
.venv\Scripts\Activate.ps1  # or source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. Run the End-to-End Demo

```bash
python demo.py
```

This single command:
1. Ingests authentic NetFlow flows from `data/sample_nf_v2.csv`.
2. Loads the registered pre-trained E-GraphSAGE model from `models/egraphsage/v0.1.0/`.
3. Constructs the communication graph window (nodes = IP endpoints, edges = communication flows).
4. Runs GNN edge inference and flags attacks.
5. Solves the counterfactual minimal cut to isolate threats while preserving normal business operations.
6. Prints the complete containment decision and action directives.

---

## Datasets

- **Bundled Sample:** `data/sample_nf_v2.csv` (1.2 MB, 5,022 stratified authentic NetFlow flows across all classes) is included for immediate testing.
- **Full Benchmark Datasets:** To train on full datasets (e.g., NF-UNSW-NB15-v2 or NF-UNSW-NB15-v3):
  1. Download the CSV from [University of Queensland NetFlow Datasets](https://staff.itee.uq.edu.au/mwr/datasets.html).
  2. Place it in `data/raw/` (e.g., `data/raw/NF-UNSW-NB15-v3.csv`).
  3. Update `config/detection.yaml` with the dataset path.

See [data/README.md](data/README.md) for further dataset information.

---

## Verification & Testing

Run the automated test suite and code quality checks:

```bash
# Run unit tests
pytest tests/unit -v

# Run lint checks
ruff check graphguard tests

# Run type checks
mypy --strict graphguard/core
```

Or using the Makefile:
```bash
make test
make lint
make typecheck
make p0  # runs lint + typecheck + tests
```

---

## Model Training & Evaluation

### Train E-GraphSAGE
To train a model on your dataset and automatically register it into the model registry:
```bash
python -m graphguard.detect.train
```

### Run Containment Benchmarking
To benchmark the CVC counterfactual cut solver against traditional baseline strategies (No-Response, Isolate-Host, Block-All):
```bash
python -m graphguard.eval.containment_eval
```
Results and trade-off frontier plots are output to `eval/results/`.

---

## Deployment Infrastructure (Optional)

GraphGuard includes Docker Compose recipes for enterprise message streaming and telemetry stores:
- **Redpanda / Kafka:** Distributed NetFlow ingestion bus
- **TimescaleDB / PostgreSQL:** Graph window time-series and incident storage
- **Redis:** Distributed containment locking and active rule state

```bash
# Copy local development environment configuration
cp .env.example .env

# Spin up services
docker compose -f deploy/docker-compose.yml --env-file .env up -d

# Check service health
docker compose -f deploy/docker-compose.yml --env-file .env ps
```

---

## Project Structure

```
GraphGuard/
├── config/              # YAML configuration for assets, policies, & detection
│   ├── action_policy.yaml
│   ├── assets.yaml
│   └── detection.yaml
├── data/                # NetFlow datasets
│   ├── README.md
│   └── sample_nf_v2.csv # Authentic sample dataset (included)
├── demo.py              # 1-command quickstart entry point
├── deploy/              # Docker Compose deployment manifests
├── docs/                # Architecture, design decisions, and specifications
├── eval/                # Evaluation output and Pareto frontier plots
├── graphguard/          # Core package
│   ├── core/            # Models, interfaces, and configurations
│   ├── graph/           # Graph window builders & feature extraction
│   ├── detect/          # E-GraphSAGE GNN model & training pipeline
│   ├── containment/     # Counterfactual minimal cut solver & cost evaluator
│   ├── enforce/         # Dry-run enforcer & rule reconciliation
│   └── eval/            # Containment benchmark suite
├── models/              # Versioned pre-trained model registry
├── requirements.txt     # Standard pip requirements
├── requirements-dev.txt # Development requirements
├── pyproject.toml       # PEP 621 package metadata & configuration
└── tests/               # Comprehensive unit and integration test suite
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
