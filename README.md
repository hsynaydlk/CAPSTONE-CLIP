# CAPSTONE-CLIP

This repository contains the CAPSTONE-CLIP experiment and evaluation workflows.

## Included workflows

- `scripts/run_capstone_benchmark.py` runs CAPSTONE-CLIP, comparison methods, control experiments, retrieval tests, generalization tests, efficiency measurements, and data audits.
- `scripts/run_protocol_baselines.py` runs the protocol-matched AVION-PM and APPLeNet-PM baselines with the same data splits.

## Setup

Use Python 3.10 or newer. Install a CUDA-compatible PyTorch build first, then install the remaining packages:

```bash
pip install -r requirements.txt
```

Copy `configs/example.env` values into your environment and update the dataset paths. The default paths are compatible with Google Colab and Google Drive.

## Main benchmark

```bash
export CAPSTONE_RUN_PROFILE=full
python scripts/run_capstone_benchmark.py
```

Available profiles:

- `smoke`: Minimal execution check.
- `core`: Main classification and retrieval experiments.
- `full`: Complete benchmark, controls, generalization, ablations, diagnostics, and efficiency tests.

## Protocol-matched baselines

```bash
export CAPSTONE_BASELINE_MODE=full
export CAPSTONE_REFERENCE_RUN=outputs_capstone_clip_full_v1
python scripts/run_protocol_baselines.py
```

The baseline workflow verifies the CAPSTONE split manifest before running.

## Data layout

Place the scene-classification and caption datasets under `CAPSTONE_DATA_ROOT`. Feature caches and run outputs are stored under `CAPSTONE_ROOT`.

## Outputs

Each workflow saves raw results, aggregate metrics, figures, configuration files, protocol metadata, and a ZIP archive of the run directory.
