# DFL

DFL is a lightweight framework for **decentralized federated learning (DFL)**, built for academic research. In DFL there is no central server. Clients sit on a communication graph, train locally, and exchange models with their neighbors through gossip. The framework runs the whole process as a simulation in a single program. It provides simple abstract base classes for clients and a coordinator that handles the topology, model exchange, evaluation and logging. A new DFL algorithm only needs a client class. Nothing in it targets deployment (no Docker, networking or RPC), so the code stays small and the work stays on the algorithm.

The framework also supports **dynamic membership**: some clients can join the network partway through training. This lets you study late-joining clients, cold-start and network expansion, in addition to the standard static setting.

## Features

- **Simplicity**: one abstract `Client` class with four methods to implement. The coordinator handles graph construction, neighbor exchange, evaluation and learning-rate scheduling.
- **Lightweight**: a pure PyTorch simulation with no industrial components.
- **Flexible topologies**: random, ring or fully connected graphs, either static or regenerated every round (gossip). Connectivity is checked automatically. The spectral gap (λ) of the mixing matrix can be computed exactly or estimated.
- **Dynamic membership**: you control which clients join late, when they join, and how arrival times are distributed (`single`, `uniform`, `even`, `normal`, or `none`).
- **Heterogeneous data**: Dirichlet, label-based, cluster-based and fixed-shard Non-IID splits.
- **Network-level metrics**: besides per-client accuracy, precision, recall, F1 and loss, the framework logs the consensus error Ω, the step size η, λ̂ and the stationarity gradient norm ‖∇f(θ̄)‖² every round.
- **Reproducible experiments**: structured per-run logs, an idempotent batch launcher that skips completed runs, and an aggregation script that produces tables and figures.

### Built-in Methods

| `--fl_method` | Description |
|---|---|
| `dfedavg` | Decentralized FedAvg (neighbor averaging) |
| `dfedsam` | DFedSAM (sharpness-aware local training) |
| `dfedpgp` | DFedPGP (personalized partial-model exchange) |
| `dfedmtkd` / `dfedmtkdrl` | Multi-teacher knowledge distillation variants |
| `dfedcad` | DKM-clustering with alignment and distillation |
| `fedgo`, `qfedcg`, `retfhd` | Additional baselines |
| `wc` | Warm start + calibrated step size for late joiners |
| `localonly`, `ellocal` | Local-training-only references |

**Datasets:** MNIST, EMNIST, CIFAR-10, CIFAR-100, Tiny-ImageNet
**Models:** `lenet`, `cnn`, `leafcnn1`, `alexnet`, `vgg16`, `resnet18`, `resnet18gn`, `resnet50`, `tinyvit`

## Getting Started

### Prerequisites

Install Python 3 and the required libraries:

```bash
pip install torch torchvision torcheval numpy pandas matplotlib tqdm scikit-learn
```

MNIST, EMNIST and the CIFAR datasets download automatically into `./data` on first use. Tiny-ImageNet must be placed manually at `./data/tiny-imagenet-200/`.

### Generating Heterogeneous Datasets

Split a dataset across 100 clients with a Dirichlet distribution:

```bash
python build_statistical_heterogeneity.py --dataset_name cifar10 --clients_num 100 --split_method dirichlet --alpha 0.4 --seed 42 --dataset_indexes_dir client_indices
```

The command saves the per-client train and validation indices to `client_indices/<dataset>_<split>_<alpha>/client_<id>/`. Other split methods are `label`, `clusters`, `even` and `dirfix<N>`. `dirfix<N>` uses Dirichlet label proportions with a fixed shard size per client (`--shard_size`), so you can vary the number of clients while keeping the work per round constant.

### Running DFL Experiments

Run DFedAvg on CIFAR-10. Client 81 joins at round 25 and the other clients are present from the start:

```bash
python main.py --fl_method dfedavg --dataset_name cifar10 --alpha 0.4 --model lenet --optimizer_name adam --lr 1e-3 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --set_single_delay_client 81 --seed 42 --device cuda --split_method dirichlet
```

For a standard static DFL run with no late joiners, add `--temp_client_dist none`.

### Parameters

**General training**
- `--fl_method`: The DFL method to use (e.g., `dfedavg`).
- `--dataset_name`: Name of the dataset (e.g., `cifar10`).
- `--split_method`, `--alpha`: The data split to load, which must already be generated.
- `--model`: Model to train (e.g., `lenet`).
- `--optimizer_name`: Optimizer (`sgd`, `adam`, `adamw`).
- `--lr`: Learning rate for the client model.
- `--batch_size`: Batch size for training.
- `--local_epochs`: Number of local epochs per round.
- `--n_rounds`: Number of communication rounds.
- `--scheduler_name`: Learning-rate scheduler (`reduce_on_plateau`, `cosine_annealing`, `multi_step`, `constant`, etc.).
- `--augment`: Set to `1` to enable standard training-time augmentation.
- `--seed`: Seed for random number generation.
- `--device`: Device to run the training on (`cpu`, `cuda`, `mps`).

**Topology**
- `--topology`: Communication graph family (`random`, `ring`, `full`).
- `--num_conn`: Number of neighbors per client (random graphs).
- `--gossip`: Set to `1` to regenerate the graph every round, or `0` for a fixed graph.
- `--symmetry`: Set to a non-zero value for an undirected (symmetric) graph.
- `--lambda_estimator`: How λ is obtained (`exact`, `power5`, `power10`, `power20`, `surrogate`).

**Dynamic membership**
- `--temp_client_dist`: Distribution of late joiners (`single`, `uniform`, `even`, `normal`, `none`).
- `--delay_client_ratio`: Fraction of clients that join late.
- `--set_single_delay_client`: The client ID that joins late in `single` mode (`-1` picks one at random).
- `--minimum_join_rounds`: The earliest round at which a late client can join.
- `--join_round_max`: The upper bound of the arrival window (`-1` means `n_rounds`).

**Logging**
- `--exp_group`, `--run_name`, `--results_dir`: Where the run is saved (`results/<exp_group>/<run_name>/`).
- `--eval_every`: Number of rounds between stationarity gradient-norm evaluations.

Method-specific hyperparameters (`--lambda_kd`, `--rho`, `--n_clusters`, the `wc_*` options and others) are documented in `utils/args.py`.

### Implementing a New Method

Subclass `clients.client.Client` and implement four methods:

```python
from copy import deepcopy
from clients.client import Client

class MyClient(Client):
    def set_init_model(self, model):   # called when the client starts or joins
        self.model = deepcopy(model)
        if self.neighbor_model_weights:   # a late joiner may receive neighbor models first
            self.aggregate()

    def train(self):                    # local update for one round
        self._local_train()

    def send_model(self):               # what this client shares with its neighbors
        return self.model.state_dict()

    def aggregate(self):                # combine the neighbor models in self.neighbor_model_weights
        self.model.load_state_dict(self._weight_aggregation())
        self.neighbor_model_weights.clear()
```

Then register the class in `clients/client_factory.py` and add its name to the `--fl_method` choices in `utils/args.py`. A method that needs a non-standard exchange protocol can supply its own coordinator exchange routine; `Coordinator.interchange_model_dfedpgp` is an example.

### Batch Experiments

`run_experiments.py` defines experiment groups as grids of runs. Runs that have already finished are skipped, so you can interrupt and resume at any time:

```bash
python run_experiments.py --list                        # list experiment groups
DFL_DEVICE=cuda python run_experiments.py <group_name>  # run one group
python run_experiments.py <group_name> --dry_run        # print the commands only
```

### Analyzing Results

Each run writes the following files to `results/<exp_group>/<run_name>/`:
- `config.json`: the run configuration.
- `metrics.jsonl`: per-round overall, per-client and network-level metrics.
- `events.jsonl`: events such as client joins.
- `summary.json`: final metrics.

Aggregate all runs into tables and figures:

```bash
python -m analysis.aggregate --results_dir results --out results/report
```

This produces `runs_summary.csv`, a mean ± std table for each group, and accuracy and consensus-error curves. It also computes derived metrics such as the accuracy dip after a client joins and the number of rounds needed to recover. The notebooks in `visualization/` contain further plotting examples.

## Project Structure

```
main.py                             entry point for a single run
coordinator.py                      simulation engine: topology, exchange, joins, evaluation
build_statistical_heterogeneity.py  Non-IID data splitting
run_experiments.py                  idempotent batch launcher
clients/client.py                   abstract client base class
clients/client_factory.py           method registry
clients/dfl_method_clients/         built-in method implementations
models/                             model zoo (CNNs, ResNets, VGG, TinyViT, DKM layer)
utils/                              arguments, data/model loading, logging, event bus
analysis/                           result aggregation and figure export
```

## Contributing

We welcome contributions to improve the framework! Please submit pull requests or open an issue to discuss your ideas.

## Acknowledgements and References

- The `leafcnn1` model follows the FEMNIST CNN from [LEAF](https://github.com/TalwalkarLab/leaf).
