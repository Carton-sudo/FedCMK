# FedCMK

PyTorch implementation of prototype-guided classifier training and residual decision correction for federated learning under label distribution skew.

## Installation

Use Python 3.11 and install the dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Choose a PyTorch build appropriate for your CPU or accelerator. Use `--device cpu`, `--device mps`, or `--device cuda` explicitly when launching a run.

## Prepare data

```bash
python scripts/prepare_data.py --dataset cifar10
```

The downloader supports MNIST, CIFAR-10 and CIFAR-100 (`--dataset all` downloads all three). Data are downloaded only when this command is run. Dataset terms remain those of their providers.

## Train

```bash
python run.py --method FedCMK --scene c10-s2 --seed 2022 --device cpu
```

Add `--verbose` to display per-round progress. The default mode displays start and completion messages only. A new run writes its own configuration, metrics and configured model artifacts under `runs/`. Existing output directories are protected against accidental overwrite.

```bash
python run.py --list --method FedCMK
python run.py --method FedCMK --scene c10-s2 --seed 2022 --dry-run
python run.py --method FedCMK --scene c10-s2 --seed 2022 --device cuda --output runs/my_run
```

`--dry-run` resolves inputs without training. `--config` accepts a specific JSON from `configs/index.csv`. `--data-root` changes the dataset location. Configuration paths are relative to this repository, not the current working directory.

The supported scenes are `m-s2`, `m-lda01`, `c10-s2`, `c10-s3`, `c10-s5`, `c10-s10`, `c10-lda005`, `c10-lda01`, `c10-lda03`, `c10-lda05`, `c100-s10` and `c100-lda01`.

## Methods and configurations

The algorithm registry exposes only FedCMK, FedAvg, FedProx, MOON, FedNTD and FedLMD. FedCMK uses a client-private EMA teacher, class-quality weighting during training, and no class-quality weighting in the inference residual.

`configs/benchmark/` contains standard comparisons, and `configs/resources/` contains participation/local-epoch configurations. Each JSON specifies the training budget, hyperparameters, device and storage policy. These configurations use 400 communication rounds.

```bash
python run.py --suite resources --method FedCMK --scene c10-s2 --seed 2022 --participants 50 --local-epochs 5 --device cpu
```

FedCMK source is in `src/algorithms/fedcmk/`. Its `_shared/` directory contains private loss, model-splitting, prototype-training and optional diagnostic helpers, not separately registered algorithms. No historical development algorithms or ablation methods are exposed.

## Batch execution

```bash
python scripts/run_suite.py --suite benchmark --method FedCMK --scene c10-s2 --device cpu
```

This prints commands only. Add `--execute` to run them sequentially. `--method available` selects all included methods. Add `--seed` to limit the selection.

## Outputs and storage

Training creates `round_metrics.csv` and run metadata. Checkpoint creation follows the selected configuration. To disable model-bearing outputs for a new run, use `--save-model-artifacts false`; such a run cannot be resumed. To resume, pass a trusted checkpoint inside the same output directory using `--resume` and the matching `--output`. Never load an untrusted checkpoint.

W&B networking is disabled by the entry point; no account or API key is required. Optional diagnostic source modules are included as runtime dependencies; they contain no saved diagnostic measurements.

## Repository contents

```text
run.py              Configuration selection and training entry
requirements.txt    Runtime dependencies
src/                Algorithms, losses, models, data loading and runtime
configs/            Input configurations and their index
assets/             Dataset partition indices and client-selection schedules
scripts/            Dataset download and batch-launch helpers
LICENSES/           Retained third-party license text
THIRD_PARTY_NOTICES.md
```

Partition files contain dataset indices, and schedules contain simulated client selections. Neither contains images, trained weights or performance measurements. Shared implementation is organized under the FedCMK package.

No pretrained models, saved results, plots, manuscript files, test outputs or internal audit records are bundled. Metrics are produced by running the code. `data/`, `runs/`, checkpoints and logs are ignored by Git.

## Attribution and licensing

The runtime builds on the FedLMD and FedNTD implementations; see `THIRD_PARTY_NOTICES.md` and `LICENSES/`. The retained upstream MIT notice applies to the corresponding inherited code. No blanket license is assigned here to author/project additions; the owners must settle that license before public release.
